# import os, sys, json, math, itertools
import pandas as pd, numpy as np
# import warnings

# from tqdm import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch

from PACE.utils import sample, generate_steps, compute_uot_plans, get_batch, set_seed
from PACE.losses import MMD_loss, OT_loss1, OT_loss2
from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint as odeint2
from PACE.models import  ODEFunc2


def pretrain(model, df, optimizer, steps, n_epoch=1000, test_interval=100,
    batch_size = 256,
    hold_one_out=False,
    hold_out='random',
    logger=None,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    reg = 0.01,
    reg_m = [10,np.inf],
    norm_cost=False,
    stage = 'pretrain'
):
    print('relative_mass',relative_mass)
    time_labels = df['samples'].to_numpy()
    data = df.iloc[:,1:].to_numpy()

    X_selected = [data[time_labels == v] for t, v in enumerate(np.unique(time_labels)) if v != hold_out]
    t_train = [v for v in np.unique(time_labels) if v != hold_out]
    print('t_train:', t_train)
    logger.info('Begin flow and growth matching')
    uot_plans = compute_uot_plans(X_selected, steps, reg=reg, reg_m=reg_m, draw=False, norm_cost=norm_cost)
    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    for i in progress_bar:
        optimizer.zero_grad()
        t, tt, xt, ut, gt = get_batch(X_selected, steps, batch_size, uot_plans, relative_mass, device)
        vt, gt_pred = model(t,tt,xt,stage)
        vloss = torch.mean((vt - ut)**2)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=0.1)
        # torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=0.1)

        optimizer.step()
        logging.info(f"Epoch {i}: loss={loss.item():.6f}, vloss={vloss.item():.6f}, gloss={gloss.item():.6f}")
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}","vloss": f"{vloss.item():.6f}", "gloss": f"{gloss.item():.6f}"})
    
    return model, vloss_list, gloss_list, loss_list



#%%
def train(
    model, df, groups, steps, optimizer, n_epochs=20, 
    criterion=None,
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    device=None,
    logger=None,

    relative_mass=None,
    initial_size=None,
    best_model_path=None,
    stepsize=0.01,
    n_big_batch=1
):

    steps = steps
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    # Send model to cuda and specify it as training mode
    if use_cuda:
        model = model.cuda()
    
    model.train()
    model.to(device)

    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')
    for epoch in tqdm(range(n_epochs)):
        
        # apply local loss
        if local_loss and not global_loss:
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device) #torch.Size([n_sample_size, 1])
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            data_t0.to(device)
            # print('sample size:',sample_size)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] 
                steps = generate_steps(groups)
                print(steps)
            for step_idx, (t0, t1) in enumerate(steps):
                print(f'epoch_{epoch}_{step_idx}:')
                logger.info(f'epoch_{epoch}_{step_idx}:')                      
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(np.round((1/n_big_batch)*(df[df['samples']==t1].values.shape[0])).astype(int),)
                data_t1 = sample(df, t1, size=size1, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                time.to(device)
                print(time)
                relative_mass_now = relative_mass[t1]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0)
                t=time.to(device)#(t0,t1)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t=odeint(ODEFunc2(model,torch.tensor([t0,t1]).float().expand(data_t0.shape[0],2).to(device)), initial_state_energy, t, options=dict(step_size=stepsize), method='euler')
                lnw_t_last = lnw_t[-1]
                mu = torch.exp(lnw_t_last)
                print(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                print(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                logger.info(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                logger.info(f'diff_mass:{mu.detach().cpu().numpy().sum()-relative_mass_now}')
                logger.info(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
               
                loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
             
            
                print(f'Otloss:{loss_ot.item():.6}')
                logger.info(f'Otloss:{loss_ot.item():.6}')
                
                loss=loss_ot
                print(f'total loss:{loss.item():.6}')
                logger.info(f'total loss:{loss.item():.6}')
        

                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Check if the current Otloss is the new minimum
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                # Save the model's state_dict
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)          
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses




def weighted_gene_mean(x, w):
    return (w[:, None] * x).sum(dim=0)

def delta_cosine_similarity(delta_pred, delta_true, eps=1e-8):
    num = torch.dot(delta_pred, delta_true)
    denom = (delta_pred.norm() * delta_true.norm() + eps)
    return num / denom

def sign_agreement(delta_pred, delta_true):
    return (torch.sign(delta_pred) == torch.sign(delta_true)).float().mean()

def delta_pearson(delta_pred, delta_true):
    vx = delta_pred - delta_pred.mean()
    vy = delta_true - delta_true.mean()
    return (vx * vy).mean() / (vx.std() * vy.std() + 1e-8)


from collections import defaultdict
def test(
    model, df, groups, steps,optimizer, n_epochs=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    device=None,
    logger=None,

    relative_mass=None,
    initial_size=None,
    best_model_path=None,
    stepsize=0.01,
    n_big_batch=1,
    test = False,
    seed = 42
):
    metrics = defaultdict(lambda: {
        "RME": [],
        "OT": [],
        "PCC": [],
        "DeltaCos": [],
        "DeltaSign": [],
        "DeltaPCC": []
    })
    set_seed(seed)
    time_labels = df['samples'].to_numpy()
    time2idx = {t: k for k, t in enumerate(np.unique(time_labels))}
    idx2time = {v: k for k, v in time2idx.items()}
    print(time2idx)

    steps = steps
    print(groups)
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    # Send model to cuda and specify it as training mode
    # if use_cuda:
    #     model = model.cuda()
    
    model.eval()
    model.to(device)

    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')
    for epoch in tqdm(range(n_epochs)):
        
        # apply local loss
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device) #torch.Size([n_sample_size, 1])
            data_t0 = sample(df, idx2time[0], size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda,device=device)
            data_t0.to(device)
            # print('sample size:',sample_size)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used
            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] 
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):
                print(f'epoch_{epoch}_{step_idx}:')
                logger.info(f'epoch_{epoch}_{step_idx}:')                      
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(np.round((1/n_big_batch)*(df[df['samples']==t1].values.shape[0])).astype(int),)
                data_t1 = sample(df, t1, size=size1, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda,device = device)
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).to(device) if use_cuda else torch.Tensor([t0, t1])
                time.to(device)
                print(time)
                relative_mass_now = relative_mass[t1]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0)
                t=time.to(device)#(t0,t1)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t=odeint(ODEFunc2(model,torch.tensor([t0,t1]).float().expand(data_t0.shape[0],2).to(device)), initial_state_energy, t, options=dict(step_size=stepsize), method='euler')
                lnw_t_last = lnw_t[-1]
                mu = torch.exp(lnw_t_last)
                print(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                print(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                logger.info(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                logger.info(f'diff_mass:{mu.detach().cpu().numpy().sum()-relative_mass_now}')
                rme=np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now
                logger.info(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1).to(device)
                if test:
                    loss_ot, pcc_ot  = criterion(x_t[-1], data_t1, mu, nu,device=device, pcc_metrics = True)
                    print(f'pcc_ot:{pcc_ot.item():.6}')
                    logger.info(f'pcc_ot:{pcc_ot.item():.6}')
                else :
                    loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
                # loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
                # i_mass=i_mass+1
             
                # lnw0=lnw_t_last.detach()
                # data_t0=x_t[-1].detach()
            
                print(f'Otloss:{loss_ot.item():.6}')
                logger.info(f'Otloss:{loss_ot.item():.6}')
                
                loss=loss_ot
                print(f'total loss:{loss.item():.6}')
                logger.info(f'total loss:{loss.item():.6}')
                key = f"{t0}:{t1}"
                _mu = torch.ones(data_t0.shape[0],1).squeeze(1).to(device)
                _mu = _mu / _mu.sum()
                # t0 gene mean
                mean_t0 = weighted_gene_mean(data_t0, _mu)

                # predicted t1 gene mean
                mean_pred = weighted_gene_mean(x_t[-1], mu)

                # true t1 gene mean
                mean_true = weighted_gene_mean(data_t1, nu)

                delta_pred = mean_pred - mean_t0
                delta_true = mean_true - mean_t0
                delta_cos = delta_cosine_similarity(delta_pred, delta_true)
                delta_sign = sign_agreement(delta_pred, delta_true)
                delta_pcc  = delta_pearson(delta_pred, delta_true)


                metrics[key]["RME"].append(rme)
                metrics[key]["OT"].append(loss_ot.item())
                metrics[key]["PCC"].append(pcc_ot.item())
                metrics[key]["DeltaCos"].append(delta_cos.item())
                metrics[key]["DeltaSign"].append(delta_sign.item())
                metrics[key]["DeltaPCC"].append(delta_pcc.item())
                # apply local loss as we calculate it
                # if apply_losses_in_time and local_loss:
                #     loss.backward()
                #     optimizer.step()
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Check if the current Otloss is the new minimum
            # if current_ot_loss < min_ot_loss:
            #     min_ot_loss = current_ot_loss
            #     # Save the model's state_dict
            #     torch.save(model.state_dict(), best_model_path)
            #     print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.to(device)
            
            # if not apply_losses_in_time:
            #     batch_loss.backward()
            #     optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)          
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    summary = {}
    for k, v in metrics.items():
        summary[k] = {
            "RME": (np.mean(v["RME"]), np.std(v["RME"])),
            "OT":  (np.mean(v["OT"]),  np.std(v["OT"])),
            "PCC": (np.mean(v["PCC"]), np.std(v["PCC"])),
            "DeltaCos": (np.mean(v["DeltaCos"]), np.std(v["DeltaCos"])),
            "DeltaSign":  (np.mean(v["DeltaSign"]),  np.std(v["DeltaSign"])),
            "DeltaPCC": (np.mean(v["DeltaPCC"]), np.std(v["DeltaPCC"]))
        }
    return summary