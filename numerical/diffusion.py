import torch.utils.data
import scipy.signal as sig
import numpy as np
from network import *
from utils import plot_driver_generation, plot_station_generation, plot_training_loss
from maxsam import *
from tqdm import tqdm
import math

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class DDPM:
    def __init__(self, opt, data_loader):
        super().__init__()
        if opt.network == "attention":
            self.eps_model = Attention(opt).to(opt.device)
        else:
            self.eps_model = CNN(opt).to(opt.device)
        self.opt = opt
        self.n_steps = opt.n_steps
        if opt.schedule == "linear":
            self.beta = torch.linspace(opt.beta_start, opt.beta_end, opt.n_steps, device=opt.device)
        elif opt.schedule == "cosine":
            self.beta = self.cosine_beta_schedule(opt.n_steps, opt.beta_end)
        elif opt.schedule == "quadratic":
            self.beta = torch.linspace(opt.beta_start**0.5, opt.beta_end**0.5, opt.n_steps, device=opt.device)**2
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # self.sigma2 = self.beta
        self.sigma2 = torch.cat((torch.tensor([self.beta[0]], device=opt.device), self.beta[1:]*(1-self.alpha_bar[0:-1])/(1-self.alpha_bar[1:])))
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.data_loader = data_loader
        self.loss_func = nn.MSELoss()
        p1, p2 = int(0.75 * opt.n_epochs), int(0.9 * opt.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)

    def cosine_beta_schedule(self, n_steps, beta_end=0.02):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.linspace(0, 1, n_steps + 1)
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2)**2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(beta, 0, 0.999).to(device)




    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        # print("alpha_bar range:", self.alpha_bar.min().item(), self.alpha_bar.max().item())
        mean = (alpha_bar**0.5)*x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5)*eps

    def p_sample(self, xt, c, t):
        eps_theta = self.eps_model(xt, c, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha)/(1 - alpha_bar)**0.5
        mean = (xt - eps_coef*eps_theta)/(alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5)*z

    def predict_x0(self, xt, c, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha_bar = torch.clamp(alpha_bar, min=1e-8)

        eps_theta = self.eps_model(xt, c, t)
        
        x0_raw = (xt - (1 - alpha_bar).sqrt() * eps_theta) / alpha_bar.sqrt()
        
        # x0 = torch.clamp(x0_raw, 0.0, 1.0)
        return x0_raw
    
    # qpth version
    def opt_loss1(self, x0, x_recon):
        from qpth.qp import QPFunction
        batch_size, dim = x0.shape
        torch.manual_seed(42)
        np.random.seed(100)

        P = np.diag(np.ones(dim))

        y = np.random.randn(dim)
        q = -2 * y
        # A = np.random.randn(1, dim)
        # #A = np.ones((1,n))
        # b = np.array([1])

        # G = np.vstack((np.eye(dim),-np.eye(dim)))
        # h = np.vstack((np.ones(dim),np.zeros(dim))).reshape(2*dim)
        m, p = 10, 5
        A = np.random.randn(m, dim)
        b = np.random.randn(m)

        G = np.random.randn(p, dim)
        h = np.random.randn(p)

        P_tch = torch.from_numpy(P)
        y_tch = torch.from_numpy(y).requires_grad_()
        q_tch = torch.from_numpy(q).requires_grad_()
        A_tch = torch.from_numpy(A)
        b_tch = torch.from_numpy(b)
        G_tch = torch.from_numpy(G)
        h_tch = torch.from_numpy(h)
        P1 = P_tch.unsqueeze(0).expand(batch_size, P_tch.size(0), P_tch.size(1)).to(x0.device)
        q1 = q_tch.unsqueeze(0).expand(batch_size, q_tch.size(0)).to(x0.device)
        A1 = A_tch.unsqueeze(0).expand(batch_size, A_tch.size(0), A_tch.size(1)).to(x0.device)
        b1 = b_tch.unsqueeze(0).expand(batch_size, b_tch.size(0)).to(x0.device)
        G1 = G_tch.unsqueeze(0).expand(batch_size, G_tch.size(0), G_tch.size(1)).to(x0.device)
        h1 = h_tch.unsqueeze(0).expand(batch_size, h_tch.size(0)).to(x0.device)

        save_dir = '/home/sun1321/src/diff2sp/Diffgen/weights_20/'
        torch.save(P_tch, save_dir + 'P.pth')
        torch.save(q_tch, save_dir + 'q.pth')
        torch.save(A_tch, save_dir + 'A.pth')
        torch.save(b_tch, save_dir + 'b.pth')
        torch.save(G_tch, save_dir + 'G.pth')
        torch.save(h_tch, save_dir + 'h.pth')
        print("save done")
        
        q0 = x0 
        q_recon = x_recon

        # solve the opt problem y(x0)
        y0 = -QPFunction(verbose=False)(P1, q0.double(), G1, h1, A1, b1)
        y_recon = -QPFunction(verbose=False)(P1, q_recon.double(), G1, h1, A1, b1)
        assert torch.isfinite(y_recon).all(), "NaN/Inf in x_recon"
        # print(f"y0 mean: {y0.mean().item():.4f}, y_recon mean: {y_recon.mean().item():.4f}")
        # loss compute
        loss_new_term = torch.norm(y0 - y_recon, p=2)  
        return loss_new_term

    def cal_loss(self, x0, c):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # keep noise on same device as x0
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)

        # predict noise for ALL timesteps/features
        eps_theta = self.eps_model(xt, c, t)           # (B, L, D)
        loss_noise = self.loss_func(noise, eps_theta)

        # optional reconstruction loss
        x_recon = self.predict_x0(xt, c, t)            # (B, L, D)
        loss_recon = self.loss_func(x0, x_recon)

        # optional boundary penalty (if you use it)
        boundary_penalty = torch.mean(torch.relu(-x_recon) + torch.relu(x_recon - 1))

        # optional opt loss: only compute if you really enable it
        lambda_opt = getattr(self.opt, "lambda_opt", 0.0)
        if lambda_opt > 0:
            loss_opt = self.opt_loss1(x0.reshape(batch_size, -1), x_recon.reshape(batch_size, -1))
        else:
            loss_opt = x0.new_tensor(0.0)

        lambda_recon = getattr(self.opt, "lambda_recon", 0.0)
        lambda_bound = getattr(self.opt, "lambda_bound", 0.0)

        loss = loss_noise + lambda_recon * loss_recon + lambda_opt * loss_opt + lambda_bound * boundary_penalty
        return loss_noise, loss_recon, loss_opt, loss


    def sample(self, weight_path, n_samples, condition):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c = torch.from_numpy(condition).type(torch.float32)
        c = c.view(1, -1).to(self.opt.device)
        with torch.no_grad():
            weight = torch.load(weight_path, map_location=self.opt.device)
            self.eps_model.load_state_dict(weight)
            self.eps_model.eval()
            # if self.opt.level == "station":
            #     if condition[0] == 1:
            #         max_sampler = pmax_sample("ACN-data/caltech/station/2019", n_samples)
            #     else:
            #         max_sampler = pmax_sample("ACN-data/jpl/station/2019", n_samples)
            # else:
            #     max_sampler = cmax_sample("ACN-data/jpl/driver", n_samples)
            # max_sampler = cmax_sample("data", n_samples)
            max_sampler = []
            for i in range(n_samples):
                if i % 3 == 0:  
                    max_sampler.append(np.random.uniform(0.8, 1.2))  
                elif i % 3 == 1:
                    max_sampler.append(np.random.normal(1.0, 0.1))  
                else:
                    max_sampler.append(np.random.exponential(0.2) + 0.8)  

            max_sampler = np.clip(max_sampler, 0.8, 1.2).tolist()
            # print(len(self.alpha_bar))
            # print(self.n_steps)
            assert len(self.alpha_bar) == self.n_steps, "Error: alpha_bar length does not match n_steps"
            
            # Collect normalized samples (typically in [0,1]) and optionally
            # de-normalized samples (original units) using saved min/max.
            gendata_norm = []
            gendata_denorm = []

            data_min = getattr(self.opt, "data_min", None)
            data_max = getattr(self.opt, "data_max", None)
            want_denorm = bool(getattr(self.opt, "sample_return_denorm", True)) and data_min is not None and data_max is not None
            if want_denorm:
                data_min = np.asarray(data_min, dtype=np.float32)
                data_max = np.asarray(data_max, dtype=np.float32)
                if data_min.shape[0] != self.opt.input_dim or data_max.shape[0] != self.opt.input_dim:
                    raise ValueError(
                        f"De-normalization stats dim mismatch: data_min/data_max have {data_min.shape[0]} cols, "
                        f"but opt.input_dim={self.opt.input_dim}."
                    )
                denom = np.maximum(data_max - data_min, getattr(self.opt, "norm_eps", 1e-8)).astype(np.float32)
            for i in range(n_samples):
                x = torch.randn([1, self.opt.seq_len, self.opt.input_dim], device=device)
                # for j in range(0, self.n_steps, 1):
                #     t = torch.ones(1, dtype=torch.long, device=device)*(self.n_steps-j-1)
                #     x = self.p_sample(x, c, t)
                for j in range(self.n_steps):
                    t_val = self.n_steps - j - 1
                    t = torch.tensor([t_val], device=device)
                    assert t_val < self.n_steps, f"t={t_val} is out of range"
                    x = self.p_sample(x, c, t)
                
                # Smoothly map output into [0,1] if requested (no hard clamp)
                act = getattr(self.opt, "sample_activation", "sigmoid")
                if act == "sigmoid":
                    x_out = torch.sigmoid(x)
                elif act == "none":
                    x_out = x
                else:
                    raise ValueError(f"Unknown sample_activation: {act} (use 'sigmoid' or 'none')")

                x_norm = x_out.squeeze().detach().cpu().numpy().astype(np.float32)  # (L, D)
                gendata_norm.append(x_norm)

                if want_denorm:
                    # Inverse min-max: x = x_norm*(max-min) + min
                    x_denorm = x_norm * denom + data_min
                    gendata_denorm.append(x_denorm)

            return {
                "norm": gendata_norm,
                "denorm": gendata_denorm if want_denorm else None,
            }
        
                # path = f"generation/{self.opt.model_name}/{self.opt.network}/{self.opt.level}/{self.opt.cond_flag}/{i}"
                # if self.opt.level == "station":
                #     gen_filt = self.station_postprocess(gen)
                #     plot_station_generation(gen_filt, path)
                # else:
                #     gen_filt1, gen_filt2 = self.driver_postprocess(gen)
                #     plot_driver_generation(gen_filt1, gen_filt2, path)

    def driver_postprocess(self, x):
        x = sig.medfilt(x, kernel_size=5)
        low_index = np.where(x < 0)[0]
        high_index = np.where(x > 32)[0]
        x[low_index], x[high_index] = 0.0, 
        x_filt1 = x
        # identify zero padding
        try:
            zero_index = np.where(x < 0.5)[0]
            invalid_index = zero_index[np.where(zero_index > 50)[0][0]]
            x_filt2 = x[0:invalid_index+1]
        except:
            x_filt2 = x_filt1
        return x_filt1, x_filt2

    def station_postprocess(self, x):
        # x[np.where(x < 0)[0]] = 0.0
        # x = sig.medfilt(x, kernel_size=5)
        # try:
        #     low_index = np.where(x < 10)[0][-1]
        #     if low_index > 200:
        #         x[low_index:] = 0
        # except:
        #     pass
        return x

    def train(self):
        epoch_loss = []
        loss1_list = [] 
        loss2_list = []
        loss3_list = []
        for epoch in range(self.opt.n_epochs):
            # pbar = tqdm(total=len(self.data_loader), desc=f'Epoch {epoch + 1}/{self.opt.n_epochs}')
            batch_loss = []
            for i, data in enumerate(self.data_loader):
                # if self.opt.level == "station":
                #     x0 = data["power"].to(self.opt.device)
                # else:
                #     x0 = data["current"].to(self.opt.device)
                x0 = data["features"].to(self.opt.device)
                # Ensure (B, L, D) for LSTM/Transformer
                if x0.dim() == 2:
                    expected = self.opt.seq_len * self.opt.input_dim
                    if x0.size(1) != expected:
                        raise ValueError(
                            f"Your data has {x0.size(1)} feature columns per row, but "
                            f"seq_len*input_dim = {self.opt.seq_len}*{self.opt.input_dim} = {expected}. "
                            "Either change opt.seq_len/opt.input_dim, or reshape your CSV layout."
                        )
                    x0 = x0.view(-1, self.opt.seq_len, self.opt.input_dim)
                c = data["labels"].to(self.opt.device)
                self.optimizer.zero_grad()
                loss1, loss2, loss3, loss = self.cal_loss(x0, c)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
                loss3_list.append(loss3.item())
                avg_loss = sum(batch_loss) / len(batch_loss)
                avg_loss1 = sum(loss1_list) / len(loss1_list)
                avg_loss2 = sum(loss2_list) / len(loss2_list)
                avg_loss3 = sum(loss3_list) / len(loss3_list)
                # pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                # pbar.update(1)
        
            epoch_loss.append(np.mean(batch_loss))

            
            print(f"epoch={epoch + 1}/{self.opt.n_epochs}, loss_noise={avg_loss1:.4f}, loss_recon={avg_loss2:.4f}, loss_opt={avg_loss3:.4f}, loss={epoch_loss[-1]}")
            self.lr_scheduler.step()

            
        save_path = f"/home/sun1321/src/diff2sp_new/output_model/epoch{epoch+1}.pt"

        torch.save(self.eps_model.state_dict(), save_path)
            
        # pbar.close()  
        plot_training_loss(epoch_loss, model_name=f"{self.opt.model_name}_{self.opt.network}_{self.opt.level}", labels=["Diffusion Loss"])