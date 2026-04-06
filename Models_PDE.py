"""
Author: AmirPouya Hemmasian (a.pouyahemmasian@gmail.com) (ahemmasi@andrew.cmu.edu)
Modified: PDE parameter conditioning via FiLM (Feature-wise Linear Modulation).
          A 2-layer MLP maps per-trajectory PDE parameters to per-channel scale (gamma)
          and shift (beta) factors that modulate the spectral output of each layer.
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
            in_dim: int,
            out_dim: int,
            proj_dim: int,
            out_vars=None,
            device=Device,
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
            source_projector: nn.Module,
            transfer_in_vars: str = 'all',
            transfer_in_vars_scalers: str = None,
            transfer_out_vars: str = 'all',
            transfer_out_vars_scalers: str = None
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
            in_vars: str = None,
            out_vars: str = None,
            trainable: bool = True
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
    """
    def __init__(
            self,
            dim: int,
            factor: int = 2,
            n_layers: int = 2,
            device=Device,
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


class PDEConditioner(nn.Module):
    """
    2-layer MLP that maps PDE parameters to FiLM scale (gamma) and shift (beta) factors.

    Input:  (B, param_dim)  — per-trajectory PDE parameters (e.g. viscosity, Reynolds number)
    Output: (B, 2 * width)  — first half is gamma delta, second half is beta

    The modulation applied in the spectral layer is:
        out = out * (1 + gamma) + beta
    where gamma and beta are zero-initialized, making the layer start as an identity.
    This is critical for stable fine-tuning when loading pretrained weights.
    """
    def __init__(self, param_dim: int, width: int, device=Device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, width, device=device),
            nn.ReLU(),
            nn.Linear(width, 2 * width, device=device),
        )
        #nn.init.zeros_(self.net[-1].weight)
        #nn.init.zeros_(self.net[-1].bias)

    def forward(self, params):
        return self.net(params)


class Factorized_Spectral_Layer_PDE(nn.Module):
    """
    Factorized Spectral Convolution Layer with FiLM conditioning on PDE parameters.

    After summing the per-dimension spectral outputs, PDE parameters are used to
    compute per-channel scale and shift factors (via PDEConditioner) that modulate
    the features before the feedforward network.
    """
    def __init__(
            self,
            width: int,
            fourier_modes,
            param_dim: int,
            share_fourier=False,
            ff_factor=2,
            ff_n_layers=2,
            device=Device
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
            weight = torch.FloatTensor(width, width, n_modes, 2).to(device)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)
            if share_fourier:
                assert all([fourier_modes[0] == m for m in fourier_modes]), \
                    'number of fourier modes should be the same across all axes for them to share'
                break

        self.feedforward = FeedForward(dim=width, factor=ff_factor, n_layers=ff_n_layers, device=device)
        self.conditioner = PDEConditioner(param_dim=param_dim, width=width, device=device)

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
        out_ft_shape[i] = shape[i] // 2 + 1
        out_ft = ft.new_zeros(*out_ft_shape)
        slicer = [slice(None)] * len(shape)
        slicer[i] = slice(n_modes)
        out_ft[slicer] = self.complex_matmul(
            input=ft[slicer],
            weight=self.fourier_weight[dim if not self.share_fourier else 0],
            spatial_dim=dim
        )
        return torch.fft.irfft(out_ft, n=shape[i], dim=i, norm='ortho')

    def forward(self, x, pde_params):
        """
        x:          (B, width, *spatial_dims)
        pde_params: (B, param_dim)
        """
        outs = [self.forward_fourier_dim(x, dim) for dim in range(self.n_spatial_dims)]
        outs = torch.stack(outs).sum(dim=0)

        film = self.conditioner(pde_params)                        # (B, 2*width)
        gamma, beta = film[:, :self.width], film[:, self.width:]   # (B, width) each
        for _ in range(self.n_spatial_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        outs = outs * (1 + gamma) + beta

        return self.feedforward(outs)

    def transfer_from(
            self,
            source_layer: nn.Module,
            fourier: bool = True,
            scaler: float = None,
            ff: bool = True,
            conditioner: bool = False,
            ):
        assert source_layer.n_spatial_dims in [1, self.n_spatial_dims], \
            'source should be either 1D or same dims'
        if scaler is None:
            scaler = 1.0

        if fourier:
            assert len(source_layer.fourier_weight) in [1, len(self.fourier_weight)], \
                'number of source fourier weights should be the same or 1'
            for i in range(len(self.fourier_weight)):
                j = 0 if len(source_layer.fourier_weight) == 1 else i
                self.fourier_weight[i].data = source_layer.fourier_weight[j].data * scaler

        if ff:
            self.feedforward.load_state_dict(source_layer.feedforward.state_dict())

        if conditioner and hasattr(source_layer, 'conditioner'):
            self.conditioner.load_state_dict(source_layer.conditioner.state_dict())

    def set_trainability(
            self,
            fourier: bool = True,
            ff: bool = True,
            conditioner: bool = True,
            trainable: bool = True
            ):
        if fourier:
            self.fourier_weight.requires_grad_(trainable)
        if ff:
            self.feedforward.requires_grad_(trainable)
        if conditioner:
            self.conditioner.requires_grad_(trainable)


class FFNO_PDE(nn.Module):
    """
    FFNO with per-layer FiLM conditioning on PDE parameters.

    Accepts pde_params of shape (B, param_dim) alongside the state dict in forward().
    The param_dim is determined by how many PDE parameters are provided at training time
    (e.g. 1 for a single viscosity value, 2 for viscosity + Reynolds number, etc.).
    """
    def __init__(
            self,
            in_vars,
            out_vars,
            in_dim: int,
            out_dim: int,
            n_layers: int,
            width: int,
            fourier_modes: tuple,
            param_dim: int,
            share_fourier=False,
            ff_n_layers=2,
            ff_factor=2,
            device=Device
            ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.fourier_modes = fourier_modes
        self.param_dim = param_dim
        self.device = device

        self.projector = Projector(
            in_vars=in_vars,
            out_vars=out_vars,
            in_dim=in_dim,
            out_dim=out_dim,
            proj_dim=width,
            device=device,
        )
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            Factorized_Spectral_Layer_PDE(
                width=width,
                fourier_modes=fourier_modes,
                param_dim=param_dim,
                share_fourier=share_fourier,
                ff_factor=ff_factor,
                ff_n_layers=ff_n_layers,
                device=device
            ) for _ in range(n_layers)
        ])

    def forward(self, x, pde_params):
        """
        x:          dict of {var: (B, in_dim, *spatial_dims)}
        pde_params: (B, param_dim)
        """
        x = self.projector.in_proj(x)
        for layer in self.layers:
            b = layer(x, pde_params)
            x = x + b
        b = self.projector.out_proj(b)
        return b

    def transfer_from(
            self,
            source_ffno: nn.Module,

            transfer_in_vars: str = 'all',
            transfer_in_vars_scalers: str = None,
            transfer_out_vars: str = 'all',
            transfer_out_vars_scalers: str = None,

            transfer_fourier_layers: str = 'all',
            transfer_fourier_scalers: str = None,
            transfer_ff_layers: str = 'all',
            transfer_conditioner_layers: str = '',
    ):
        assert self.n_layers == source_ffno.n_layers, 'number of spectral layers should be the same'
        source_ffno.to(self.device)

        self.projector.transfer_from(
            source_projector=source_ffno.projector,
            transfer_in_vars=transfer_in_vars,
            transfer_in_vars_scalers=transfer_in_vars_scalers,
            transfer_out_vars=transfer_out_vars,
            transfer_out_vars_scalers=transfer_out_vars_scalers
        )

        transfer_fourier_layers = parse_csv(transfer_fourier_layers, full=range(self.n_layers), func=int)
        transfer_fourier_scalers = parse_csv_scalers(transfer_fourier_scalers)
        if len(transfer_fourier_scalers) == 1:
            transfer_fourier_scalers = transfer_fourier_scalers * len(transfer_fourier_layers)
        elif len(transfer_fourier_scalers) != len(transfer_fourier_layers):
            raise ValueError('number of scalers should be the same as number of layers')

        transfer_ff_layers = parse_csv(transfer_ff_layers, full=range(self.n_layers), func=int)
        transfer_conditioner_layers = parse_csv(transfer_conditioner_layers, full=range(self.n_layers), func=int)

        for i in range(self.n_layers):
            scaler = transfer_fourier_scalers[transfer_fourier_layers.index(i)] if i in transfer_fourier_layers else None
            self.layers[i].transfer_from(
                source_layer=source_ffno.layers[i],
                fourier=i in transfer_fourier_layers,
                scaler=scaler,
                ff=i in transfer_ff_layers,
                conditioner=i in transfer_conditioner_layers
            )

    def set_trainability(
            self,
            in_vars: str = 'all',
            out_vars: str = 'all',
            fourier_layers: str = 'all',
            ff_layers: str = 'all',
            conditioner_layers: str = 'all',
            trainable: bool = True
            ):
        self.projector.set_trainability(
            in_vars=in_vars,
            out_vars=out_vars,
            trainable=trainable
        )

        fourier_layers = parse_csv(fourier_layers, full=range(self.n_layers), func=lambda x: int(x) % self.n_layers)
        ff_layers = parse_csv(ff_layers, full=range(self.n_layers), func=lambda x: int(x) % self.n_layers)
        conditioner_layers = parse_csv(conditioner_layers, full=range(self.n_layers), func=lambda x: int(x) % self.n_layers)

        for i in range(self.n_layers):
            self.layers[i].set_trainability(
                fourier=i in fourier_layers,
                ff=i in ff_layers,
                conditioner=i in conditioner_layers,
                trainable=trainable
            )
