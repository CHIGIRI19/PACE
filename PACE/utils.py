# %% Adapt from MIOFlow
import numpy as np, pandas as pd
import torch
import random

def group_extract(df, group, index='samples', groupby='samples'):
    
    return df.groupby(groupby).get_group(group).set_index(index).values

def sample(data, group, size=(100, ), replace=False, to_torch=False, use_cuda=False, use_mps=False, device = None):
    sub = group_extract(data, group)
    idx = np.arange(sub.shape[0])
    sampled = sub[np.random.choice(idx, size=size, replace=replace)]
    if to_torch:
        sampled = torch.Tensor(sampled).float()
        if use_cuda:
            sampled = sampled.to(device)
        if use_mps:
            sampled = sampled.mps()
    return sampled

def to_np(data):
    return data.detach().cpu().numpy()

def generate_steps(groups):
    return list(zip(groups[:-1], groups[1:]))

def generate_multi_steps(groups):
    groups = sorted(groups)
    start = groups[0]
    return [(start, g) for g in groups[1:]]
# %%
# define lookup variables
from .losses import MMD_loss, OT_loss1, OT_loss2
from PACE.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR

_valid_datasets = {
    'file': lambda file: np.load(file),
}

_valid_criterions = {
    'mmd': MMD_loss,
    'ot1': OT_loss1,
    'ot2': OT_loss2,
}

import argparse
import sys


# Define the parser
parser = argparse.ArgumentParser(prog='PACE Training', description='Train PACE')

# NOTE: Dataset specification
parser.add_argument(
    '--dataset', '-d', type=str, choices=list(_valid_datasets.keys()), required=True,
    help=(
        'Dataset of the experiment to use. '
        'If value is fullpath to a file then tries to load file. '
        'Note, if using your own file we assume it is a pandas '
        'dataframe which has a column named `samples` that correspond to '
        'the timepoints.'
    )
)

parser.add_argument(
    '--time-col', '-tc', type=str, choices='simulation_i step_ix sim_time'.split(), required=False,
    help='Time column of the dataset to use.'
)

# NOTE: Experiment specification
parser.add_argument(
    '--name', '-n', type=str, required=True, default=None,
    help='Name of the experiment. If none is provided timestamp is used.'
)

parser.add_argument(
    '--output-dir', '-od', type=str, default=RES_DIR,
    help='Where experiments should be saved. The results directory will automatically be generated here.'
)

# NOTE: Train arguments
parser.add_argument(
    '--criterion', '-c', type=str, choices=list(_valid_criterions.keys()), 
    default='mmd', required=True,
    help='a loss function, either `"mmd"` or `"emd"`. Defaults to `"mmd"`.'
)

parser.add_argument(
    '--sample-with-replacement', '-swr', type=bool, 
    action=argparse.BooleanOptionalAction, default=False,     
    help='Whether or not to sample with replacement. Defaults to `True`.'
)

parser.add_argument(
    '--hold-one-out', '-hoo', 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Whether or not to randomly hold one time pair e.g. t_1 to t_2 out when computing the global loss.'
)

parser.add_argument(
    '--hold_out', '-ho', type=int, default=-1,
    help='Defaults to `-1`. Which time point to hold out when calculating the global loss.'
)

parser.add_argument(
    '--apply-losses-in-time', '-it', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Applies the losses and does back propagation as soon as a loss is calculated. See notes for more detail.'
)

parser.add_argument(
    '--learning_rate1', '-lr1', type=float, default=1e-3,
    help='Learning rate for the training without L_ot. Defaults to `1e-3`.'
)

parser.add_argument(
    '--epochs1', '-e1', type=int, default=2000,
    help='Number of epochs without L_ot. Defaults to `2000`.'
)

parser.add_argument(
    '--norm_cost', '-nc', 
    action=argparse.BooleanOptionalAction, default=False,
    help='Whether or not to normalize the cost matrix when calculating the OT plan. Defaults to `False`.'
)

parser.add_argument(
    '--reg', '-r', type=float, default=0.03,
    help='Regularization strength for the Sinkhorn algorithm. Defaults to `0.03`.'
)

parser.add_argument(
    '--reg_m', '-rm', type=float, default=10,
    help='Mass regularization strength for the Unbalanced OT Sinkhorn algorithm. Defaults to `10`.'
)

parser.add_argument(
    '--n_big_batches', '-nbb', type=int, default=1,
    help='Number of big batches to use when training. Defaults to `1`.'
)

parser.add_argument(
    '--n_batch', '-bs', type=int, default=256,
    help='Batch size to use when training. Defaults to `256`.'
)

parser.add_argument(
    '--learning_rate2', '-lr2', type=float, default=1e-4,
    help='Learning rate for the training with L_ot. Defaults to `1e-4`.'
)

parser.add_argument(
    '--epochs2', '-e2', type=int, default=50,
    help='Number of epochs with L_ot. Defaults to `30`.'
)


import numpy as np
import torch
import ot
from torchdyn.core import NeuralODE
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import pickle

###############################preparation##############################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Big batch ---
def split_X_into_subsets(X_selected, n_parts=5, seed=42):
    np.random.seed(seed)
    split_X = []
    for t_data in X_selected:
        shuffled = np.random.permutation(t_data)
        splits = np.array_split(shuffled, n_parts)
        split_X.append(splits)
    return [list(subset) for subset in zip(*split_X)]

# --- cache UOT plan ---
def compute_and_cache_uot_plans(X_subsets, t_train, reg, reg_m, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    print("Caching UOT plans to:", cache_dir)
    for i, X_sub in enumerate(X_subsets):
        cache_file = os.path.join(cache_dir, f"uot_plan_subset_{i}.pkl")
        if os.path.exists(cache_file):
            print(f"Cache found for subset {i}, skipping...")
            continue
        print(f"Computing UOT plans for subset {i}...")
        uot_plans = compute_uot_plans(X_sub, t_train, reg=reg, reg_m=reg_m, draw=False)
        with open(cache_file, 'wb') as f:
            pickle.dump(uot_plans, f)

def load_cached_uot_plans(cache_dir, subset_idx):
    cache_file = os.path.join(cache_dir, f"uot_plan_subset_{subset_idx}.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"No cached UOT plan found for subset {subset_idx}")
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def compute_uot_plans(X,steps,reg=0.05,reg_m=[50, np.inf],draw=False, norm_cost=False):# X_selected
    uot_plans = []

    for (i,j) in tqdm(steps, desc="Computing UOT plans..."):
        X_source = X[i]
        X_target = X[j]
        n_source, n_target = X_source.shape[0], X_target.shape[0] 
        cost_matrix = ot.dist(X_source, X_target)
        if norm_cost:
            cost_matrix = cost_matrix / cost_matrix.max()

        a = np.ones(n_source) 
        b = np.ones(n_target) 

        G = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, reg, reg_m)
        uot_plans.append(G)
        
        if draw:
            source_pred = G.sum(1)
            tar_pred = G.sum(0)
            
            fig = plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.plot(a, label = f'source_true_{i}')
            plt.plot(source_pred, label = f'source_pred_{i}')
            plt.legend()

            plt.subplot(132)
            plt.plot(b, label = f'target_true_{i+1}')
            plt.plot(tar_pred, label = f'target_pred_{i+1}')
            plt.legend()
            
            plt.subplot(133)
            plt.scatter(X_source[:,0],X_source[:,1],s=source_pred, alpha=0.5)
            plt.show()

    return uot_plans

    
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0],1)], 1))

###############################train##############################################
def compute_cond_flow(x0: torch.Tensor, x1: torch.Tensor, delta_t) -> torch.Tensor:
    # Compute the conditional flow (velocity) between two states over time delta_t
    return (x1 - x0) / delta_t


def compute_cond_g(uot_plan: np.ndarray, ratio: float, idx: np.ndarray, delta_t) -> torch.Tensor:
    # Select the rows of the UOT plan corresponding to the sampled batch
    selected_uot_plan = uot_plan[idx, :]
    # Compute source weights by summing transport mass along the target dimension
    source_weights = torch.tensor(selected_uot_plan.sum(axis=-1, keepdims=True), dtype=torch.float32)
    eps = 1e-10  # small constant to avoid log(0)
    # Compute the growth term g = (log(m_t+Δt) - log(m_t)) / Δt
    return (torch.log(source_weights + eps) - torch.log(torch.ones_like(source_weights) + eps)) / delta_t


import numpy as np

def sample_map(pi: np.ndarray, batch_size: int = 256, replace: bool = True):
    # Compute the probability distribution over source indices (rows)
    row_sums = pi.sum(axis=1)
    total_sum = row_sums.sum()
    row_probs = row_sums / total_sum  # normalized probability for each source index i

    # Sample source indices i according to their row probabilities
    i_samples = np.random.choice(pi.shape[0], p=row_probs, size=batch_size, replace=replace)

    # For each sampled i, sample a target index j according to the row distribution
    j_samples = np.zeros(batch_size, dtype=int)
    for idx, i in enumerate(i_samples):
        # Normalize the row to obtain probabilities for targets j
        row_p = pi[i] / row_sums[i]
        j_samples[idx] = np.random.choice(pi.shape[1], p=row_p)

    return i_samples, j_samples


def sample_from_ot_plan(ot_plan: np.ndarray, x0: torch.Tensor, x1: torch.Tensor, batch_size: int = 256):
    i, j = sample_map(ot_plan, batch_size, replace=False)
    return x0[i], x1[j], i  # only return i is enough


def get_batch(X, steps, batch_size, uot_plans, ratios, device):
    ts = []
    tt = []
    xts = []
    uts = []
    gts = []
    
    for k, (i,j) in enumerate(steps): 
        

        uot_plan = uot_plans[k]
        ratio = ratios[i]
        
        x0, x1, idx = sample_from_ot_plan(uot_plan, X[i], X[j], batch_size)
        
        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)
        
        delta_t = j-i
        t_samp = delta_t*torch.rand(x0.shape[0], 1).type_as(x0)
        xt_samp = (t_samp * x0 + (delta_t - t_samp) * x1)/delta_t + 0.1*torch.randn_like(x0)
        ut_samp = compute_cond_flow(x0, x1,delta_t).float().to(device)
        gt_samp = compute_cond_g(uot_plan, ratio, idx, delta_t).float().to(device)
        
        ts.append(t_samp + i)
        tt.append(torch.tensor([i,j]).float().expand(x0.shape[0],2).to(device))
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
    
    return torch.cat(ts), torch.cat(tt), torch.cat(xts), torch.cat(uts), torch.cat(gts)
 

