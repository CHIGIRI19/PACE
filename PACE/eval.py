import torch, numpy as np
from PACE.utils import sample, to_np
from torchdiffeq import odeint_adjoint as odeint
from PACE.models import  PACE, ODEFunc2

# Adapt from MIOFLOW
def generate_points(
    model, df, steps, n_points=100, 
    sample_with_replacement=False, use_cuda=False, 
    samples_key='samples', sample_time=None, autoencoder=None, recon=False, device=None,
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `PACE.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_points (int): Number of points to generate.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        sample_time (list | None): Defaults to `None`. If `None` uses the group numbers in order as the 
            timepoints as specified in the column `df[samples_key]`.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        generated (float[float[]]): a list with shape `(len(sample_time), n_points, len(df.columns) - 1)`
            of the generated points.
    '''
    to_torch = True if use_cuda else False

    groups = sorted(df[samples_key].unique())
    if sample_time is None:
        sample_time = groups
    data_t0 = sample(
        df, np.min(groups), size=(n_points, ), 
        replace=sample_with_replacement, to_torch=to_torch, use_cuda=use_cuda
    )
    lnw0 = torch.log(torch.ones(data_t0.shape[0]) / data_t0.shape[0]).to(device)
    if autoencoder is not None and recon:
        data_t0 = autoencoder.encoder(data_t0)
    all_generated = []
    for step_idx, (t0, t1) in enumerate(steps):
        time = torch.tensor([t0, t1], dtype=torch.float32)
        print(time)
        if isinstance(data_t0, np.ndarray):
            data_t0 = torch.from_numpy(data_t0)
            data_t0=data_t0.float()
        print(type(data_t0),data_t0.shape)
        initial_state_energy = (data_t0, lnw0)
        generated, generate_weights=odeint(ODEFunc2(model,torch.tensor([t0,t1]).float().expand(data_t0.shape[0],2)), initial_state_energy, time)
        if autoencoder is not None and recon:
            generated = autoencoder.decoder(generated)
        if step_idx ==0:
            all_generated.append(generated[0])
            all_generated.append(generated[1])
        else:
            all_generated.append(generated[1])
    generated = torch.stack(all_generated, dim=0)
    return to_np(generated) ,to_np(torch.exp(generate_weights))

def generate_trajectories(
    model, df, n_trajectories=30, n_bins=100, 
    sample_with_replacement=False, use_cuda=False, samples_key='samples',autoencoder=None, recon=False
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `PACE.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_trajectories (int): Number of trajectories to generate.
        n_bins (int): Number of bins to use for the trajectories. More makes it smoother. Defaults to `100`.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        trajectories (float[float[]]): a list with shape `(n_bins, n_points, len(df.columns) - 1)`
            of the generated trajectories.
    '''
    groups = sorted(df[samples_key].unique())
    sample_time = np.linspace(np.min(groups), np.max(groups), n_bins)
    trajectories,trajectories_weights = generate_points(model, df, n_trajectories, sample_with_replacement, use_cuda, samples_key, sample_time,autoencoder=autoencoder, recon=recon)
    return trajectories,trajectories_weights

def generate_plot_data(
    model, df, steps, n_points, n_trajectories, n_bins, 
    sample_with_replacement=False, use_cuda=False, samples_key='samples',
    logger=None, autoencoder=None, recon=False
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `PACE.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_points (int): Number of points to generate.
        n_trajectories (int): Number of trajectories to generate.
        n_bins (int): Number of bins to use for the trajectories. More makes it smoother. Defaults to `100`.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        points (float[float[]]): a list with shape `(len(df[sample_key].unique()), n_points, len(df.columns) - 1)`
            of the generated points.
        trajectories (float[float[]]): a list with shape `(n_bins, n_points, len(df.columns) - 1)`
            of the generated trajectories.
    '''
    if logger: logger.info(f'Generating points')
    points,points_weights= generate_points(model, df, steps, n_points, sample_with_replacement, use_cuda, samples_key, None, autoencoder=autoencoder, recon=recon)
    # if logger: logger.info(f'Generating trajectories')
    # trajectories,trajectories_weights = generate_trajectories(model, df, n_trajectories, n_bins, sample_with_replacement, use_cuda, samples_key, autoencoder=autoencoder, recon=recon)
    # return points, trajectories, points_weights, trajectories_weights
    return points