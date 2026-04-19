"""
Author: AmirPouya Hemmasian (a.pouyahemmasian@gmail.com) (ahemmasi@andrew.cmu.edu)

Modification: PDE parameter conditioning and timestep conditioning are handled
by two separate MLPs (ParamMLP and TimeMLP) so that each can be independently
frozen or fine-tuned during transfer learning.
"""
from einops import rearrange
import torch
from torch import nn
from utils_train import parse_csv, parse_csv_scalers

Device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Linear(nn.Linear):
    """
    Point-Wise Linear Layer for PDE data 
    where the feature/channel dimension is the second dimension (after batch)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        x = super().forward(x)
        x = rearrange(x, 'b ... c -> b c ...')
        return x


class Projector(nn.Module):
    """
    A simple linear projector for input and output variables.

    the input/output space is a dictionary of input/output variables, each with shape (B, C, ...) where ... are spatial dimensions
    the projection space is a single tensor of shape (B, C, ...) where ... are spatial dimensions
    """
    def __init__(
            self,
            in_vars,
            in_dim : int,
            out_dim : int,
            proj_dim : int,
            out_vars = None,
            device = Device,
    ):
        self.device = device
        super().__init__()
        self.in_vars = list(in_vars)
        self.out_vars = list(out_vars) or self.in_vars
        self.in_projector = nn.ModuleDict({
            var: Linear(in_dim, proj_dim, device=device)
            for var in self.in_vars
        })
        self.out_projector = nn.ModuleDict({
            var: Linear(proj_dim, out_dim, device=device)
            for var in self.out_vars
        })

    def in_proj(self, x: dict) -> torch.FloatTensor:
        return torch.stack([self.in_projector[var](x[var]) for var in x]).sum(dim=0)

    def out_proj(self, x: torch.FloatTensor) -> dict:
        return {var: self.out_projector[var](x) for var in self.out_vars}
    
    def transfer_from(
            self,
            source_projector : nn.Module,
            transfer_in_vars : str = 'all',
            transfer_in_vars_scalers : str = None,
            transfer_out_vars : str = 'all',
            transfer_out_vars_scalers : str = None
            ):
        transfer_parser = lambda x: list(x.split('>')) if '>' in x else x

        in_vars = parse_csv(transfer_in_vars, full=self.in_vars, func=transfer_parser)
        in_vars_scalers = parse_csv_scalers(transfer_in_vars_scalers)
        
        if len(in_vars_scalers) == 1:
            in_vars_scalers = in_vars_scalers * len(in_vars)
        elif len(in_vars_scalers) != len(in_vars):
            raise ValueError('number of scalers should be the same as number of variables')

        out_vars = parse_csv(transfer_out_vars, full=self.out_vars, func=transfer_parser)
        out_vars_scalers = parse_csv_scalers(transfer_out_vars_scalers)
        
        if len(out_vars_scalers) == 1:
            out_vars_scalers = out_vars_scalers * len(out_vars)
        elif len(out_vars_scalers) != len(out_vars):
            raise ValueError('number of scalers should be the same as number of variables')

        for i, source_target in enumerate(in_vars):
            if not isinstance(source_target, list): source_var, target_var = source_target, source_target
            else: source_var, target_var = source_target
            self.in_projector[target_var].weight.data = source_projector.in_projector[source_var].weight.data * in_vars_scalers[i]
            self.in_projector[target_var].bias.data = source_projector.in_projector[source_var].bias.data * in_vars_scalers[i]

        for i, source_target in enumerate(out_vars):
            if not isinstance(source_target, list): source_var, target_var = source_target, source_target
            else: source_var, target_var = source_target
            self.out_projector[target_var].weight.data = source_projector.out_projector[source_var].weight.data * out_vars_scalers[i]
            self.out_projector[target_var].bias.data = source_projector.out_projector[source_var].bias.data * out_vars_scalers[i]
    
    def set_trainability(
            self, 
            in_vars : str = None,
            out_vars : str = None,
            trainable : bool = True
            ):
        in_vars = parse_csv(in_vars, full=self.in_vars)
        for var in in_vars:
            self.in_projector[var].requires_grad_(trainable)

        out_vars = parse_csv(out_vars, full=self.out_vars)
        for var in out_vars:
            self.out_projector[var].requires_grad_(trainable)


class FeedForward(nn.Module):
    """
    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/feedforward.py
    Excluded LayerNorm and Dropout for simplicity.
    """
    def __init__(
            self, 
            dim: int, 
            factor: int = 2, 
            n_layers: int = 2,
            device = Device,
            ):
        super().__init__()
        self.device = device
        self.layers = []
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.extend([
                Linear(in_dim, out_dim, device=device),
                nn.ReLU(inplace=True)
            ])
        self.layers = nn.Sequential(*self.layers[:-1])

    def forward(self, x):
        return self.layers(x)
    

class ParamMLP(nn.Module):
    """
    Encodes PDE scalar parameters (viscosity, Re, etc.) → latent (B, width).

    These are physical constants fixed per trajectory. Kept separate from
    TimeMLP so it can be frozen/replaced independently during transfer learning.

    Input:  pde_params  (B, n_params)
    Output:             (B, width)
    """
    def __init__(
            self,
            n_params : int,
            width    : int,
            hidden   : int = None,
            device = Device,
            ):
        super().__init__()
        self.device = device
        hidden = hidden or 2 * width
        self.mlp = nn.Sequential(
            nn.Linear(n_params, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, width, device=device),
        )

    def forward(self, pde_params: torch.FloatTensor) -> torch.FloatTensor:
        return self.mlp(pde_params)


class TimeMLP(nn.Module):
    """
    Encodes the normalised timestep t ∈ [0, 1] → latent (B, width).

    Kept separate from ParamMLP because the timestep is a dynamic positional
    signal (changes every sample) while PDE params are fixed per trajectory.
    Separation lets you freeze this during transfer to a new PDE while only
    retraining ParamMLP, or vice-versa.

    Input:  t  (B,) or (B, 1)   normalised to [0, 1]
    Output:    (B, width)
    """
    def __init__(
            self,
            width  : int,
            hidden : int = None,
            device = Device,
            ):
        super().__init__()
        self.device = device
        hidden = hidden or 2 * width
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, width, device=device),
        )

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)   # (B,) → (B, 1)
        return self.mlp(t)


def _broadcast_latent(latent: torch.FloatTensor, field: torch.FloatTensor) -> torch.FloatTensor:
    """
    Expand a (B, C) latent to match the spatial shape of `field` (B, C, *spatial).
    """
    n_spatial = field.ndim - 2
    latent = latent.view(*latent.shape, *([1] * n_spatial))
    return latent.expand_as(field)


class Factorized_Spectral_Layer(nn.Module):
    """
    General Factorized Spectral Convolution Layer for 1D, 2D, or 3D spatial data.

    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_2d.py
    """
    def __init__(
            self, 
            width: int,
            fourier_modes,
            share_fourier = False,
            ff_factor = 2,
            ff_n_layers = 2,
            device = Device
            ):
        super().__init__()
        self.width = width
        self.fourier_modes = fourier_modes

        self.n_spatial_dims = len(fourier_modes)
        assert self.n_spatial_dims in [1, 2, 3], f'Only up to 3D supported. got {self.n_spatial_dims}D'

        if self.n_spatial_dims == 1:
            share_fourier = True
        self.share_fourier = share_fourier

        self.fourier_weight = nn.ParameterList()
        for i in range(self.n_spatial_dims):
            n_modes = fourier_modes[i]
            weight = torch.FloatTensor(width, width, n_modes, 2)
            weight = weight.to(device)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)
            if share_fourier:
                assert all([fourier_modes[0] == n_modes for n_modes in fourier_modes]), 'number of fourier modes should be the same across all axes for them to share'
                break

        self.feedforward = FeedForward(dim=width, factor=ff_factor, n_layers=ff_n_layers, device=device)
    
    def complex_matmul(self, input, weight, spatial_dim):
        in_str = 'bi' + 'xyz'[:self.n_spatial_dims]
        weight_str = 'io' + 'xyz'[spatial_dim]
        out_str = 'bo' + 'xyz'[:self.n_spatial_dims]
        return torch.einsum(
            f'{in_str},{weight_str}->{out_str}',
            input, 
            torch.view_as_complex(weight)
            )
    
    def forward_fourier_dim(self, x, dim):
        i = dim + 2
        n_modes = self.fourier_modes[dim]
        shape = x.shape

        ft = torch.fft.rfft(x, dim=i, norm='ortho')        
        out_ft_shape = list(shape)
        out_ft_shape[i] = shape[i]//2 + 1
        out_ft = ft.new_zeros(*out_ft_shape)

        slicer = [slice(None)] * len(shape)
        slicer[i] = slice(n_modes)
        
        out_ft[slicer] = self.complex_matmul(
            input = ft[slicer],
            weight = self.fourier_weight[dim if not self.share_fourier else 0],
            spatial_dim = dim
            )
        out = torch.fft.irfft(out_ft, n=shape[i], dim=i, norm='ortho')
        return out
    
    def forward(self, x):
        outs = []
        for dim in range(self.n_spatial_dims):
            out = self.forward_fourier_dim(x, dim)
            outs.append(out)
        outs = torch.stack(outs).sum(dim=0)
        outs = self.feedforward(outs)
        return outs
    
    def transfer_from(
            self, 
            source_layer : nn.Module,
            fourier : bool = True,
            scaler : float = None,
            ff : bool = True
            ):
        assert source_layer.n_spatial_dims in [1, self.n_spatial_dims], 'source should be either 1D or same dims'
        if scaler is None: scaler = 1.0

        if fourier:
            assert len(source_layer.fourier_weight) in [1, len(self.fourier_weight)], 'number of source fourier weights should be the same or 1'
            for i in range(len(self.fourier_weight)):
                j = 0 if len(source_layer.fourier_weight) == 1 else i
                self.fourier_weight[i].data = source_layer.fourier_weight[j].data * scaler

        if ff:
            self.feedforward.load_state_dict(source_layer.feedforward.state_dict())

    def set_trainability(
            self, 
            fourier : bool = True,
            ff : bool = True,
            trainable : bool = True
            ):
        if fourier:
            self.fourier_weight.requires_grad_(trainable)
        if ff:
            self.feedforward.requires_grad_(trainable)
    

class FFNO(nn.Module):
    """
    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_2d.py

    General FFNO for 1D, 2D, or 3D spatial data.

    Conditioning
    ------------
    Two independent MLPs inject information after the first encoder layer:

      ParamMLP  — physical PDE parameters (viscosity, Re, …), fixed per trajectory
                  enable with n_pde_params > 0
      TimeMLP   — normalised timestep t ∈ [0, 1], varies per sample
                  enable with use_time_mlp=True  (default)

    Both latents are broadcast over spatial dims and summed into the hidden state:

        x += broadcast( param_mlp(pde_params) )   # if enabled
        x += broadcast( time_mlp(t) )             # if enabled

    Transfer learning
    -----------------
    New PDE, same time domain:
        model.set_trainability(trainable=False, time_mlp=False, param_mlp=True)
        # also call transfer_from(..., transfer_param_mlp=False, transfer_time_mlp=True)

    Same PDE, different time window:
        model.set_trainability(trainable=False, param_mlp=False, time_mlp=True)

    Usage
    -----
    # Dataset produces t separately from pde_params now — see dataset_pde.py
    *chunks, pde_params, t = batch
    output = model(chunks[0], pde_params=pde_params, t=t)
    """
    def __init__(
            self,
            in_vars,
            out_vars,
            in_dim : int, 
            out_dim : int,
            n_layers: int,
            width: int,
            fourier_modes: tuple,
            share_fourier = False,
            ff_n_layers = 2,
            ff_factor = 2,
            # ── PDE parameter conditioning ───────────────────────────────────
            n_pde_params : int = 0,         # 0 = disabled
            param_mlp_hidden : int = None,  # hidden dim, default 2*width
            # ── Timestep conditioning ────────────────────────────────────────
            use_time_mlp : bool = True,     # False = disabled
            time_mlp_hidden : int = None,   # hidden dim, default 2*width
            # ────────────────────────────────────────────────────────────────
            device = Device
            ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.fourier_modes = fourier_modes
        self.device = device
        self.n_pde_params = n_pde_params
        self.use_time_mlp = use_time_mlp

        self.projector = Projector(
            in_vars = in_vars,
            out_vars = out_vars,
            in_dim = in_dim,
            out_dim = out_dim,
            proj_dim = width,
            device = device,
            )
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            Factorized_Spectral_Layer(
                width = width,
                fourier_modes = fourier_modes,
                share_fourier = share_fourier,
                ff_factor = ff_factor,
                ff_n_layers = ff_n_layers,
                device = device
                ) for _ in range(n_layers)
                ])

        # ── ParamMLP (PDE physics) ───────────────────────────────────────────
        self.param_mlp = ParamMLP(
            n_params = n_pde_params,
            width    = width,
            hidden   = param_mlp_hidden,
            device   = device,
        ) if n_pde_params > 0 else None

        # ── TimeMLP (temporal position) ──────────────────────────────────────
        self.time_mlp = TimeMLP(
            width  = width,
            hidden = time_mlp_hidden,
            device = device,
        ) if use_time_mlp else None
        # ────────────────────────────────────────────────────────────────────

    def forward(
            self,
            x          : dict,
            pde_params : torch.FloatTensor = None,   # (B, n_pde_params)
            t          : torch.FloatTensor = None,   # (B,) or (B, 1)
            ):
        """
        x          : {var: (B, C, *spatial)}
        pde_params : (B, n_pde_params)  required when n_pde_params > 0
        t          : (B,) or (B, 1)     normalised to [0,1], required when use_time_mlp=True
        """
        x = self.projector.in_proj(x)

        # ── layer 0 ──────────────────────────────────────────────────────────
        b = self.layers[0](x)
        x = x + b

        # ── conditioning injections (after encoder layer, before the rest) ───
        if self.param_mlp is not None:
            if pde_params is None:
                raise ValueError(
                    "FFNO was built with n_pde_params > 0 but `pde_params` "
                    "was not passed to forward()."
                )
            x = x + _broadcast_latent(self.param_mlp(pde_params), x)

        if self.time_mlp is not None:
            if t is None:
                raise ValueError(
                    "FFNO was built with use_time_mlp=True but `t` "
                    "was not passed to forward()."
                )
            x = x + _broadcast_latent(self.time_mlp(t), x)
        # ────────────────────────────────────────────────────────────────────

        for layer in self.layers[1:]:
            b = layer(x)
            x = x + b

        b = self.projector.out_proj(b)
        return b
    
    def transfer_from(
            self,
            source_ffno : nn.Module,
            transfer_in_vars : str = 'all',
            transfer_in_vars_scalers : str = None, 
            transfer_out_vars : str = 'all', 
            transfer_out_vars_scalers : str = None, 
            transfer_fourier_layers : str = 'all',
            transfer_fourier_scalers : str = None,
            transfer_ff_layers : str = 'all',
            # ── MLP transfer flags ───────────────────────────────────────────
            transfer_time_mlp  : bool = True,   # time encoding transfers well across PDEs
            transfer_param_mlp : bool = False,  # PDE-specific; usually retrain from scratch
            # ────────────────────────────────────────────────────────────────
    ):
        assert self.n_layers == source_ffno.n_layers, 'number of spectral layers should be the same'
        source_ffno.to(self.device)
        
        self.projector.transfer_from(
            source_projector = source_ffno.projector, 
            transfer_in_vars = transfer_in_vars, 
            transfer_in_vars_scalers = transfer_in_vars_scalers,
            transfer_out_vars = transfer_out_vars,
            transfer_out_vars_scalers = transfer_out_vars_scalers
            )

        transfer_fourier_layers = parse_csv(transfer_fourier_layers, full=range(self.n_layers), func=int)
        transfer_fourier_scalers = parse_csv_scalers(transfer_fourier_scalers)
        
        if len(transfer_fourier_scalers) == 1:
            transfer_fourier_scalers = transfer_fourier_scalers * len(transfer_fourier_layers)
        elif len(transfer_fourier_scalers) != len(transfer_fourier_layers):
            raise ValueError('number of scalers should be the same as number of layers')

        transfer_ff_layers = parse_csv(transfer_ff_layers, full=range(self.n_layers), func=int)
    
        for i in range(self.n_layers):
            self.layers[i].transfer_from(
                source_layer = source_ffno.layers[i], 
                fourier = i in transfer_fourier_layers,
                scaler = transfer_fourier_scalers[transfer_fourier_layers.index(i)],
                ff = i in transfer_ff_layers
                )

        # ── TimeMLP transfer ─────────────────────────────────────────────────
        if transfer_time_mlp:
            if self.time_mlp is None or source_ffno.time_mlp is None:
                raise ValueError('Both source and target must have use_time_mlp=True to transfer TimeMLP.')
            self.time_mlp.load_state_dict(source_ffno.time_mlp.state_dict())

        # ── ParamMLP transfer ────────────────────────────────────────────────
        if transfer_param_mlp:
            if self.param_mlp is None or source_ffno.param_mlp is None:
                raise ValueError('Both source and target must have n_pde_params > 0 to transfer ParamMLP.')
            if self.n_pde_params != source_ffno.n_pde_params:
                raise ValueError(
                    f'n_pde_params mismatch ({self.n_pde_params} vs {source_ffno.n_pde_params}). '
                    'Leave transfer_param_mlp=False when transferring to a different PDE.'
                )
            self.param_mlp.load_state_dict(source_ffno.param_mlp.state_dict())
        # ────────────────────────────────────────────────────────────────────
            
    def set_trainability(
            self,
            in_vars : str = 'all',
            out_vars : str = 'all',
            fourier_layers : str = 'all',
            ff_layers : str = 'all',
            trainable : bool = True,
            # ── per-MLP overrides (None = inherit from `trainable`) ──────────
            time_mlp  : bool = None,
            param_mlp : bool = None,
            ):
        """
        Fine-grained trainability control.

        Examples
        --------
        # Freeze everything, then unfreeze only ParamMLP (new PDE fine-tune):
        model.set_trainability(trainable=False, param_mlp=True)

        # Freeze only the TimeMLP, train everything else:
        model.set_trainability(trainable=True, time_mlp=False)
        """
        self.projector.set_trainability(
            in_vars = in_vars,
            out_vars = out_vars,
            trainable = trainable
            )
        
        fourier_layers = parse_csv(fourier_layers, full=range(self.n_layers), func=lambda x: int(x)%self.n_layers)
        ff_layers      = parse_csv(ff_layers,      full=range(self.n_layers), func=lambda x: int(x)%self.n_layers)

        for i in range(self.n_layers):
            self.layers[i].set_trainability(
                fourier = i in fourier_layers,
                ff = i in ff_layers,
                trainable = trainable
                )

        if self.time_mlp is not None:
            self.time_mlp.requires_grad_(trainable if time_mlp is None else time_mlp)

        if self.param_mlp is not None:
            self.param_mlp.requires_grad_(trainable if param_mlp is None else param_mlp)
