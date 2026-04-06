import os
import h5py
import yaml
import itertools
import pickle
import time
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from config_parser import parse_args, save_config
from Datasets import PDEDataset
from utils_train import set_seed, parse_csv, parse_csv_scalers, Relative_Lp_Loss, get_next_input
from Models_PDE import FFNO_PDE, Projector, Linear, FeedForward, Factorized_Spectral_Layer_PDE, PDEConditioner
from plotting import plot_experiments, plot_training, model_namer, transfer_namer


os.makedirs('./results', exist_ok=True)
Device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────────────────────────────────────
# PDE-parametrized dataset
# ──────────────────────────────────────────────────────────────────────────────

class ParametrizedPDEDataset(PDEDataset):
    """
    Extends PDEDataset to carry per-trajectory PDE parameters.

    After loading the dataset normally, call set_pde_params() with a tensor
    of shape (N, param_dim).  Each call to __getitem__ then returns a tuple
        (snapshots_list, pde_params_for_that_trajectory)
    so the DataLoader yields (xs, pde_params) where pde_params has shape (B, param_dim).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_params = None

    def set_pde_params(self, pde_params: torch.Tensor):
        assert pde_params.shape[0] == self.N, \
            f'pde_params has {pde_params.shape[0]} rows but dataset has {self.N} trajectories'
        self.pde_params = pde_params.to(self.device)

    def __getitem__(self, i):
        snapshots = super().__getitem__(i)
        traj_idx, _ = self.get_index(i)
        params = self.pde_params[self.indexes[traj_idx]]  # (param_dim,)
        return snapshots, params


def load_pde_params_from_h5(data_path: str, param_var_names: list, device=Device) -> torch.Tensor:
    """
    Load PDE parameter variables from an h5 file.
    Each named dataset must have shape (N,) or (N, k).
    Returns a tensor of shape (N, total_param_dim).
    """
    params_list = []
    with h5py.File(data_path, 'r') as f:
        for name in param_var_names:
            if name not in f:
                raise KeyError(f"PDE parameter '{name}' not found in {data_path}. "
                               f"Available keys: {list(f.keys())}")
            data = torch.as_tensor(f[name][:], dtype=torch.float32, device=device)
            if data.ndim == 1:
                data = data.unsqueeze(-1)  # (N,) → (N, 1)
            params_list.append(data)
    return torch.cat(params_list, dim=-1)  # (N, param_dim)


def make_constant_pde_params(n_trajs: int, param_values: list, device=Device) -> torch.Tensor:
    """
    Build a (N, param_dim) tensor by broadcasting constant values to all trajectories.
    """
    params = torch.tensor(param_values, dtype=torch.float32, device=device)  # (param_dim,)
    return params.unsqueeze(0).expand(n_trajs, -1).contiguous()


def get_pde_params(data_path: str, n_trajs: int, args, device=Device) -> torch.Tensor:
    """
    Resolve PDE params from args.  Priority:
        1. pde_param_vars  — load named variables from the h5 file
        2. pde_param_values — broadcast constant values
    """
    if args.pde_param_vars:
        var_names = [v.strip() for v in args.pde_param_vars.split(',') if v.strip()]
        return load_pde_params_from_h5(data_path, var_names, device=device)
    elif args.pde_param_values:
        values = [float(v.strip()) for v in args.pde_param_values.split(',') if v.strip()]
        return make_constant_pde_params(n_trajs, values, device=device)
    else:
        raise ValueError(
            'Either pde_param_vars or pde_param_values must be set in the config. '
            'pde_param_vars: comma-separated h5 variable names with shape (N,) or (N,k). '
            'pde_param_values: comma-separated constant floats applied to all trajectories.'
        )


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities (PDE-aware versions of utils_train functions)
# ──────────────────────────────────────────────────────────────────────────────

def forward_pass_loss_pde(
        model,
        xs,
        pde_params,
        loss_fn=Relative_Lp_Loss,
        loss_reduction='mean',
        keep_time=False,
        ):
    rollout = len(xs) - 1

    if rollout == 0:
        x = xs[0]
        return loss_fn(model(x, pde_params), x, reduction=loss_reduction)

    model_input = xs[0]
    losses = []
    for r in range(1, rollout + 1):
        model_output = model(model_input, pde_params)
        losses.append(loss_fn(model_output, xs[r], reduction=loss_reduction))
        if rollout == 1:
            break
        model_input = get_next_input(model_input, model_output)

    losses = torch.stack(losses)
    return losses if keep_time else losses.mean()


def train_iters_pde(
        model,

        train_dataset,
        config_train_data_for_training,
        config_train_data_for_validation,

        val_dataset,
        config_val_data_for_validation,

        optimizer=optim.Adam,
        loss_fn=Relative_Lp_Loss,
        loss_reduction='mean',
        iters=5000,
        batch_size=64,

        val_iters=None,
        val_rollouts=None,

        running_avg_window=10,
        ):

    if not val_iters:
        val_iters = [iters]
    elif isinstance(val_iters, int):
        val_iters = [val_iters]
    elif isinstance(val_iters, str):
        val_iters = parse_csv(val_iters, full=[iters], func=int)

    if not val_rollouts:
        val_rollouts = [1]
    elif isinstance(val_rollouts, int):
        val_rollouts = [val_rollouts]
    elif isinstance(val_rollouts, str):
        val_rollouts = parse_csv(val_rollouts, full=[1], func=int)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    opt = optimizer(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=50)

    training_losses = []
    val_loss_train_data = {f'r{r}': {} for r in val_rollouts}
    val_loss_val_data = {f'r{r}': {} for r in val_rollouts}
    train_times = []
    val_times = {f'r{r}': [] for r in val_rollouts}

    it = 0
    iter_pbar = tqdm(range(iters), desc='train iter ', leave=False)

    model.train()
    config_train_data_for_training(dataset=train_dataset)
    train_rollout = train_dataset.rollout

    while it < iters:

        for xs, pde_params in train_loader:
            start_time = time.time()

            some_key = list(xs[0].keys())[0]
            b = xs[0][some_key].shape[0]
            opt.zero_grad()
            loss = forward_pass_loss_pde(
                model=model,
                xs=xs,
                pde_params=pde_params,
                loss_fn=loss_fn,
                loss_reduction=loss_reduction,
            )
            loss.backward()
            opt.step()

            torch.cuda.synchronize()
            end_time = time.time()

            train_times.append(end_time - start_time)
            if loss_reduction == 'mean':
                training_losses.append(loss.item())
            elif loss_reduction == 'sum':
                training_losses.append(loss.item() / b)

            running_avg_train_loss = np.mean(training_losses[-running_avg_window:])
            scheduler.step(running_avg_train_loss)
            iter_pbar.set_postfix_str(f'running avg train loss: {running_avg_train_loss:.6f}')
            it += 1
            iter_pbar.update(1)

            if it in val_iters:
                model.eval()
                r_pbar = tqdm(val_rollouts, leave=False)
                for r in r_pbar:
                    r_pbar.set_description_str(
                        f'validation rollout ' +
                        ''.join([f'[{rr}]' if rr == r else f'({rr})' for rr in val_rollouts]) + ' '
                    )

                    config_train_data_for_validation(dataset=train_dataset)
                    train_dataset.config_autoregression(rollout=r)
                    train_loader_val = DataLoader(train_dataset, batch_size=batch_size)

                    sum_loss, sum_b = 0., 0
                    t_times = []
                    for xs_v, params_v in tqdm(train_loader_val, desc='validating on train data', leave=False):
                        bv = xs_v[0][list(xs_v[0].keys())[0]].shape[0]
                        t0 = time.time()
                        with torch.no_grad():
                            lv = forward_pass_loss_pde(model, xs_v, params_v, loss_fn, 'sum', keep_time=True)
                        torch.cuda.synchronize()
                        t_times.append(time.time() - t0)
                        sum_loss = sum_loss + lv
                        sum_b += bv
                    val_loss_train_data[f'r{r}'][f'it{it}'] = (sum_loss / sum_b).cpu().numpy()
                    val_times[f'r{r}'].extend(t_times)

                    config_val_data_for_validation(dataset=val_dataset)
                    val_dataset.config_autoregression(rollout=r)
                    val_loader_r = DataLoader(val_dataset, batch_size=batch_size)

                    sum_loss_v, sum_bv = 0., 0
                    for xs_v, params_v in tqdm(val_loader_r, desc='validating on val data', leave=False):
                        bv = xs_v[0][list(xs_v[0].keys())[0]].shape[0]
                        with torch.no_grad():
                            lv = forward_pass_loss_pde(model, xs_v, params_v, loss_fn, 'sum', keep_time=True)
                        sum_loss_v = sum_loss_v + lv
                        sum_bv += bv
                    val_loss_val_data[f'r{r}'][f'it{it}'] = (sum_loss_v / sum_bv).cpu().numpy()

                model.train()
                config_train_data_for_training(dataset=train_dataset)
                train_dataset.config_autoregression(rollout=train_rollout)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            if it >= iters:
                break

    batch_train_time = np.mean(train_times)
    batch_val_time = {f'r{r}': np.mean(val_times[f'r{r}']) for r in val_rollouts}

    return {
        'training_losses': training_losses,
        'val_loss_train_data': val_loss_train_data,
        'val_loss_val_data': val_loss_val_data,
        'batch_train_time': batch_train_time,
        'batch_val_time': batch_val_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers copied / adapted from Train_FFNO.py
# ──────────────────────────────────────────────────────────────────────────────

def is_result_or_seed_col(col_name):
    col = col_name.lower()
    return ('loss' in col) or ('time' in col) or ('seed' in col)


def save_table_to_excel(result_dir, result_table):
    result_table.to_excel(result_dir + '/table.xlsx', index=False)

    indexes = [col for col in result_table.columns if not is_result_or_seed_col(col)]
    if len(indexes) > 0:
        result_table.set_index(indexes).to_excel(result_dir + '/table_indexed.xlsx')

    non_varying_cols = [
        col for col in result_table.columns
        if result_table[col].nunique() <= 1 and not is_result_or_seed_col(col)
    ]
    if len(non_varying_cols) > 0:
        result_table_trimmed = result_table.drop(columns=non_varying_cols)
        result_table_trimmed.to_excel(result_dir + '/table_trimmed.xlsx', index=False)

        trimmed_indexes = [col for col in result_table_trimmed.columns if not is_result_or_seed_col(col)]
        if len(trimmed_indexes) > 0:
            result_table_trimmed.set_index(trimmed_indexes).to_excel(result_dir + '/table_trimmed_indexed.xlsx')


def load_data(args):
    train_path = os.path.join(args.data_dir, args.data_train)
    data_train = ParametrizedPDEDataset(
        data_path=train_path,
        t_start=args.data_train_t_start,
        t_end=args.data_train_t_end,
        dt=args.data_train_dt,
        rx=args.data_train_rx,
        verbose=True,
    )
    pde_params_train = get_pde_params(train_path, data_train.N, args, device=Device)
    data_train.set_pde_params(pde_params_train)

    if args.data_val not in [args.data_train, None, '']:
        val_path = os.path.join(args.data_dir, args.data_val)
        data_val = ParametrizedPDEDataset(
            data_path=val_path,
            t_start=args.data_val_t_start,
            t_end=args.data_val_t_end,
            dt=args.data_val_dt,
            rx=args.data_val_rx,
        )
        pde_params_val = get_pde_params(val_path, data_val.N, args, device=Device)
        data_val.set_pde_params(pde_params_val)
        val_separate = True
    else:
        data_val = data_train
        val_separate = False

    param_dim = pde_params_train.shape[1]
    return data_train, data_val, val_separate, param_dim


def set_up_data(data_train, data_val, args):
    cfg = {
        'dt': args.dt,
        'in_snapshots': args.in_snapshots,
        'out_snapshots': args.out_snapshots,
        'skip': args.skip,
        'rollout': args.rollout,
    }
    data_train.config_autoregression(**cfg)
    data_val.config_autoregression(**cfg)


def get_data_config_functions(args, val_separate):

    def config_train_data_for_training(dataset):
        dataset.config(
            subset=0.0 if val_separate else args.val_size,
            where=args.val_mode,
            reverse=True,
            seed=args.val_seed,
            frac=args.frac_train_data,
            frac_seed=args.seed,
        )

    def config_train_data_for_validation(dataset):
        dataset.config(
            subset=0.0 if val_separate else args.val_size,
            where=args.val_mode,
            reverse=True,
            seed=args.val_seed,
            frac=1.0,
            frac_seed=args.seed,
        )

    def config_val_data_for_validation(dataset):
        dataset.config(
            subset=1.0 if val_separate else args.val_size,
            where=args.val_mode,
            reverse=False,
            seed=args.val_seed,
            frac=1.0,
            frac_seed=args.seed,
        )

    return [config_train_data_for_training, config_train_data_for_validation, config_val_data_for_validation]


def train_ffno_pde(
        ffno,
        args,
        data_train,
        data_val,
        config_train_data_for_training,
        config_train_data_for_validation,
        config_val_data_for_validation,
        pretrained_ffno='',
        ):

    set_up_data(data_train, data_val, args)

    if pretrained_ffno:
        ffno.transfer_from(
            source_ffno=pretrained_ffno,

            transfer_in_vars=args.transfer_in_vars,
            transfer_in_vars_scalers=args.transfer_in_vars_scalers,
            transfer_out_vars=args.transfer_out_vars,
            transfer_out_vars_scalers=args.transfer_out_vars_scalers,

            transfer_fourier_layers=args.transfer_fourier_layers,
            transfer_fourier_scalers=args.transfer_fourier_scalers,
            transfer_ff_layers=args.transfer_ff_layers,
            transfer_conditioner_layers=args.transfer_conditioner_layers,
        )
        ffno.set_trainability(trainable=False)
        ffno.set_trainability(
            in_vars=args.tune_in_vars,
            out_vars=args.tune_out_vars,
            fourier_layers=args.tune_fourier_layers,
            ff_layers=args.tune_ff_layers,
            conditioner_layers=args.tune_conditioner_layers,
            trainable=True,
        )

    training_result = train_iters_pde(
        model=ffno,

        train_dataset=data_train,
        config_train_data_for_training=config_train_data_for_training,
        config_train_data_for_validation=config_train_data_for_validation,

        val_dataset=data_val,
        config_val_data_for_validation=config_val_data_for_validation,

        optimizer=getattr(optim, args.optimizer),
        loss_fn=Relative_Lp_Loss,
        loss_reduction='mean',
        iters=args.iters,
        batch_size=args.batch_size,

        val_iters=args.val_iters,
        val_rollouts=args.val_rollouts,
    )

    return ffno, training_result


def is_valid_model(args) -> bool:
    if args.data_ndims == 1 and not args.share_fourier:
        return False
    return True


def is_valid_transfer(args, ffno) -> bool:
    transfer_parser = lambda x: list(x.split('>')) if '>' in x else [x, x]
    transfer_in_vars = parse_csv(args.transfer_in_vars, full=ffno.projector.in_vars, func=transfer_parser)
    transfer_in_vars = [v[1] if isinstance(v, list) else v for v in transfer_in_vars]
    transfer_out_vars = parse_csv(args.transfer_out_vars, full=ffno.projector.out_vars, func=transfer_parser)
    transfer_out_vars = [v[1] if isinstance(v, list) else v for v in transfer_out_vars]
    tune_in_vars = parse_csv(args.tune_in_vars, full=ffno.projector.in_vars)
    tune_out_vars = parse_csv(args.tune_out_vars, full=ffno.projector.out_vars)

    all_layers = list(range(ffno.n_layers))
    layer_indexer = lambda x: int(x) % ffno.n_layers
    transfer_fourier_layers = parse_csv(args.transfer_fourier_layers, full=all_layers, func=layer_indexer)
    transfer_ff_layers = parse_csv(args.transfer_ff_layers, full=all_layers, func=layer_indexer)
    tune_fourier_layers = parse_csv(args.tune_fourier_layers, full=all_layers, func=layer_indexer)
    tune_ff_layers = parse_csv(args.tune_ff_layers, full=all_layers, func=layer_indexer)

    if set(tune_in_vars) != set(tune_out_vars):
        return False
    if not set(transfer_fourier_layers + transfer_ff_layers):
        return False
    if not set(tune_fourier_layers + tune_ff_layers):
        return False

    if len(tune_fourier_layers) == 1:
        if len(tune_ff_layers) > 1:
            return False
        if all_layers[-1] in tune_fourier_layers and all_layers[-1] not in tune_ff_layers:
            return False

    if len(tune_ff_layers) == 1:
        if len(tune_fourier_layers) > 1:
            return False
        if all_layers[0] in tune_ff_layers and all_layers[0] not in tune_fourier_layers:
            return False

    if not (transfer_fourier_layers == transfer_ff_layers == all_layers):
        if tune_ff_layers not in [[], all_layers]:
            return False
        if tune_fourier_layers not in [[], all_layers]:
            return False

    return True


def get_exp_combinations(exp_configs_yaml):
    if not exp_configs_yaml:
        return [], []
    with open(exp_configs_yaml) as file:
        configs = OrderedDict(yaml.safe_load(file))
    hyperparams = list(configs.keys())
    combs = [dict(zip(configs.keys(), comb)) for comb in itertools.product(*configs.values())]
    return hyperparams, combs


def find_matching_pretrained_ffno(pretrained_ffno_dir, model_args):
    pretrained_dir_result_table = pd.read_excel('/' + pretrained_ffno_dir + '/table.xlsx')
    matching_args = {
        arg: vars(model_args)[arg]
        for arg in ['in_snapshots', 'out_snapshots', 'dt', 'skip', 'n_layers', 'ffno_width', 'fourier_modes', 'seed']
    }
    matching_row = pretrained_dir_result_table[
        (pretrained_dir_result_table[matching_args.keys()] == pd.Series(matching_args)).all(axis=1)
    ]
    if len(matching_row) == 0:
        raise ValueError('No matching pretrained model found!')
    return '/content/1d_results/models/model_00000.pt'


def train_and_save_result(
        args,
        data_train,
        data_val,
        val_separate,
        param_dim,

        idx,
        result_dir,
        result_table,
        pretrained_ffno_path='',

        log_plots=True,
        ):

    set_seed(args.seed)

    ffno = FFNO_PDE(
        in_vars=data_train.Vars.keys(),
        out_vars=data_train.Vars.keys(),
        in_dim=args.in_snapshots,
        out_dim=args.out_snapshots,

        n_layers=args.n_layers,
        width=args.ffno_width,
        fourier_modes=args.data_ndims * [args.fourier_modes],
        param_dim=param_dim,
        share_fourier=args.share_fourier,

        device=Device,
    )

    print(pretrained_ffno_path)

    ffno, training_result = train_ffno_pde(
        ffno,
        args,
        data_train,
        data_val,
        *get_data_config_functions(args, val_separate),
        pretrained_ffno=torch.load(pretrained_ffno_path, weights_only=False, map_location=Device)
        if pretrained_ffno_path else None,
    )

    vars(args)['batch_train_time'] = training_result['batch_train_time']

    y_vars = []
    print(f"\"{args.val_rollouts}\"")
    for r in parse_csv(args.val_rollouts, full=[1], func=int):
        vars(args)[f'batch_val_time_r{r}'] = training_result['batch_val_time'][f'r{r}']

        for val_iter in parse_csv(args.val_iters, full=[args.iters], func=int):
            vars(args)[f'train_loss_r{r}_it{val_iter}'] = training_result['val_loss_train_data'][f'r{r}'][f'it{val_iter}'].mean()
            vars(args)[f'val_loss_r{r}_it{val_iter}'] = training_result['val_loss_val_data'][f'r{r}'][f'it{val_iter}'].mean()
            y_vars.append(f'train_loss_r{r}_it{val_iter}')
            y_vars.append(f'val_loss_r{r}_it{val_iter}')

    for key in vars(args):
        if key not in result_table.columns and ('time' in key or 'loss' in key):
            result_table[key] = ''

    result_table.loc[idx] = vars(args)
    save_table_to_excel(result_dir, result_table)

    torch.save(ffno, result_dir + f'/models/model_{idx:05d}.pt')

    with open(result_dir + f'/trainings/training_{idx:05d}.pkl', 'wb') as file:
        pickle.dump(training_result, file)

    idx += 1

    if log_plots:

        def frac_tick_formatter(val, pos=None):
            N_train = (1 - args.val_size) * data_train.N
            return f'$\\frac{{{round(val*N_train)}}}{{{round(N_train)}}}$'

        plot_experiments(
            result_df=result_table,
            result_dir=result_dir,
            group_params=['n_layers', 'ffno_width', 'fourier_modes'],
            group_namer=model_namer,
            subgroup_namer=transfer_namer,
            result_keywords=['loss', 'time'],
            random_keywords=['seed'],
            varying_x_vars_only=True,
            compact_group_name=True,
            compact_subgroup_name=False,
            all_y_in_one=False,
            x_vars=['frac_train_data'],
            x_labels=['downstream samples'],
            x_scales=['log'],
            x_tickers=[frac_tick_formatter],
            y_vars=None,
            y_label=None,
            y_scales=None,
            y_tickers=None,
            save_dir=None,
            save=True,
            show=False,
        )

        plot_training(
            result_df=result_table,
            result_dir=result_dir,
            group_params=None,
            result_keywords=['loss', 'time'],
            group_namer=model_namer,
            subgroup_namer=transfer_namer,
            compact_group_name=True,
            compact_subgroup_name=False,
            smooth_window=10,
            zoom_size=0.1,
            save_dir=None,
            save=True,
            show=False,
        )

    return result_table, idx, ffno


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    main_args, default_config = parse_args(base_config_path='PreLowD/configs_FFNO/_base_pde.yaml')
    print(main_args.model_configs)

    torch.serialization.add_safe_globals([
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ParameterList,
        FeedForward, Factorized_Spectral_Layer_PDE, PDEConditioner,
        torch.nn.modules.container.ModuleList,
        FFNO_PDE, Projector, Linear,
        torch.nn.modules.container.ModuleDict,
    ])
    torch.serialization._include_weights_only_default = False

    result_dir = './results/' + main_args.name

    if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
        print(f'Folder {result_dir} already exists and is not empty! Do you want to overwrite it?')
        if input('y/n: ') != 'y':
            return

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/figures', exist_ok=True)
    os.makedirs(result_dir + '/models', exist_ok=True)
    os.makedirs(result_dir + '/trainings', exist_ok=True)

    save_config(default_config, result_dir + '/default_config.yaml')

    model_hyperparams, model_combs = get_exp_combinations(main_args.model_configs)
    transfer_hyperparams, transfer_combs = get_exp_combinations(main_args.transfer_configs)

    result_table = pd.DataFrame(columns=model_hyperparams + ['transfer_from'] + transfer_hyperparams)

    data_train, data_val, val_separate, param_dim = load_data(main_args)
    print(f'PDE param_dim: {param_dim}')

    idx = 0
    print(f'Starting experiments for {main_args.name}')

    def descriptor(some_args, varying_params):
        parts = []
        for param in varying_params:
            cur = vars(some_args)[param]
            s = f'{param}:' + ''.join(
                [f'[{v}]' if v == cur else f'({v})' for v in varying_params[param]]
            )
            parts.append(s)
        return ' | '.join(parts)

    all_model_args = []
    varying_model_params = dict()
    for model_comb in model_combs:
        model_args = deepcopy(main_args)
        vars(model_args).update(model_comb)
        model_args.transfer_from = ''
        if is_valid_model(model_args):
            all_model_args.append(model_args)
            for k, v in model_comb.items():
                if k not in varying_model_params:
                    varying_model_params[k] = [v]
                elif v not in varying_model_params[k]:
                    varying_model_params[k].append(v)

    varying_model_params = {k: v for k, v in varying_model_params.items() if len(v) > 1}

    model_args_pbar = tqdm(all_model_args, leave=True)
    for model_args in model_args_pbar:
        model_args_pbar.set_description('model | ' + descriptor(model_args, varying_model_params) + ' ')

        result_table, idx, ffno = train_and_save_result(
            args=model_args,
            data_train=data_train,
            data_val=data_val,
            val_separate=val_separate,
            param_dim=param_dim,

            idx=idx,
            result_dir=result_dir,
            result_table=result_table,
            pretrained_ffno_path='',

            log_plots=False,
        )

        if not main_args.transfer_from:
            continue

        pretrained_ffno_match_path = find_matching_pretrained_ffno(main_args.transfer_from, model_args)
        model_args.transfer_from = pretrained_ffno_match_path

        all_transfer_args = []
        varying_transfer_params = dict()
        for transfer_comb in transfer_combs:
            transfer_args = deepcopy(model_args)
            vars(transfer_args).update(transfer_comb)
            if is_valid_transfer(transfer_args, ffno):
                all_transfer_args.append(transfer_args)
                for k, v in transfer_comb.items():
                    if k not in varying_transfer_params:
                        varying_transfer_params[k] = [v]
                    elif v not in varying_transfer_params[k]:
                        varying_transfer_params[k].append(v)

        varying_transfer_params = {k: v for k, v in varying_transfer_params.items() if len(v) > 1}

        transfer_args_pbar = tqdm(all_transfer_args, leave=False)
        for i, transfer_args in enumerate(transfer_args_pbar):
            transfer_args_pbar.set_description('transfer | ' + descriptor(transfer_args, varying_transfer_params) + ' ')

            result_table, idx, ffno = train_and_save_result(
                args=transfer_args,
                data_train=data_train,
                data_val=data_val,
                val_separate=val_separate,
                param_dim=param_dim,

                idx=idx,
                result_dir=result_dir,
                result_table=result_table,
                pretrained_ffno_path=pretrained_ffno_match_path,

                log_plots=False,
            )

    save_table_to_excel(result_dir, result_table)
    print('Finished experiments for', main_args.name, '\n')


if __name__ == '__main__':
    main()
