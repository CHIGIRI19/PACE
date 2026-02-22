import torch, torch.nn as nn

class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()


        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x, t_feat):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1) 
        state  = torch.cat((t,x),dim=1)
        #print(state)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = torch.add(layer(state),t_feat)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x

class growthNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(in_out_dim + 1, hidden_dim)

        self.net = nn.Sequential(
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
        
    def forward(self, t, x, t_feat):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  
        state  = torch.cat((t,x),dim=1)

        embed_state = torch.add(self.fc1(state),t_feat)

        return self.net(embed_state)

class PACE(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation, num_time_stamps):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.num_time_stamps = num_time_stamps

        # 为每个 time stamp 创建一个 Linear
        self.t_weights = nn.Parameter(
            torch.randn(num_time_stamps, hidden_dim)
        )

        self.fusion_weights = self.fusion_weights = nn.Sequential(
            nn.Linear(2, hidden_dim),   # 输入 (t_start, t_end) 
            nn.ReLU(),
            nn.Linear(hidden_dim, num_time_stamps*hidden_dim),
        )

        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)

    def forward(self, t, tt, z, stage="pretrain"):
        """
        stage: 'pretrain' or 'finetune'
        """

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            B = z.shape[0]
            device = z.device

            t_in = tt.float().to(device)   # [B, 2]

            # =========================
            # ① PRETRAIN：piecewise
            # =========================
            if stage == "pretrain":
                t_idx = tt[:, 0].long()   # 只用于 index

                t_feat = self.t_weights[t_idx]

            # =========================
            # ② FINETUNE：fusion
            # =========================
            elif stage == "finetune":
                
                # 所有 pretrain time pieces（与 t_in 无关）
                all_t_feat = self.t_weights.unsqueeze(0).expand(
                    B, -1, -1
                )  # [B, num_time_stamps, hidden_dim]

                B = t_in.shape[0]
                # [B, num_time_stamps, hidden_dim]

            
                # fusion net 输出
                fusion_out = self.fusion_weights(t_in)
                # [B, num_time_stamps * hidden_dim]

                fusion_out = fusion_out.view(
                    B,
                    self.num_time_stamps,
                    self.hidden_dim
                )
                # [B,  num_time_stamps, hidden_dim]

                fusion_out = torch.softmax(fusion_out, dim=1)
                # gating
                weighted = fusion_out * all_t_feat
                # [B,  num_time_stamps, hidden_dim]
                # sum timestamps
                t_feat = weighted.sum(dim=1)
                # [B, hidden_dim]

            else:
                raise ValueError(f"Unknown stage: {stage}")


            v = self.v_net(t, z, t_feat).float()
            g = self.g_net(t, z, t_feat).float()

        return v, g
    
import torch.nn as nn
class ODEFunc2(nn.Module):
    def __init__(self, f_net, time_stamp):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net
        self.time_stamp = time_stamp

    def forward(self, t, state, stage="finetune"):
        z, _ = state #data0,lnw0
        v, g = self.f_net(t, self.time_stamp, z, stage=stage)
        dz_dt = v
        dlnw_dt = g
        #w = torch.exp(lnw)
        
        return dz_dt.float(), dlnw_dt.float()


class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

