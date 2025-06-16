
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def get_named_eta_schedule(
        schedule_name,
        power,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        # power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas



class ResShiftTraining(nn.Module):
    def __init__(self, diff_model, encoder, decoder, power, etas_end, kappa, min_noise_level, T):
        super().__init__()

        self.diff_model = diff_model
        self.encoder = encoder
        self.decoder = decoder
        self.T = T

        sqrt_etas = get_named_eta_schedule(
            'exponential',
            power = power,
            num_diffusion_timesteps=T,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=None,
            )
        
        self.kappa = kappa

        self.sqrt_etas = torch.tensor(sqrt_etas).cuda()
        self.etas = torch.tensor(sqrt_etas**2).cuda()
        

    # G_0对应高清图像，px对应低清
    def forward(self, u0, u100):
        
        
        """
        Algorithm 1.
        """

        t = torch.randint(self.T, size=(u0.shape[0], ), device=u0.device)
        t0 = torch.zeros_like(t)
        noise = torch.randn_like(u0)
        f_hat = self.encoder(u0, t0) # [b, 1, 32, 32]


        # 对高清加噪, 注意由两部分，分别是y-x_0以及额外的noise
        x_t = (
            u100 + extract(self.etas, t, u0.shape) * (u0 - u100) +
            extract(self.sqrt_etas* self.kappa, t, u100.shape) * noise)
      

        # x_t_xy_px =  torch.concat([x_t, input_xy, a], dim=1)


        # 直接预测G
        pred_G = self.diff_model(x_t, t)
        # pred_G = y_0 - pred_res
        pred_u_hat = torch.sum(pred_G * f_hat, dim=1, keepdim=True).view(u0.shape[0], 1, u0.shape[2], u0.shape[3])
        pred_u = self.decoder(pred_u_hat, t0) # [b, 10, 32, 32]

        loss_res = F.mse_loss(pred_u, u100, reduction='sum')
        # 计算相对误差
        err = torch.norm(pred_u - u100, p=2) / torch.norm(u100, p=2)
        # loss = F.mse_loss(u, u_pred.squeeze(), reduction='mean')

        return loss_res, err

class ResShiftSampler(nn.Module):
    def __init__(self, diff_model, encoder, decoder, power, etas_end, kappa, min_noise_level, T):
        super().__init__()

        self.diff_model = diff_model
        self.encoder = encoder
        self.decoder = decoder
        self.T = T

        sqrt_etas = get_named_eta_schedule(
            'exponential',
            power = power,
            num_diffusion_timesteps=T,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=None,
            )
        
        self.kappa = kappa

        self.sqrt_etas = torch.tensor(sqrt_etas).cuda()
        self.etas = torch.tensor(sqrt_etas**2).cuda()
        self.etas_prev = torch.tensor(np.append(0.0, (sqrt_etas**2)[:-1])).cuda()
        self.alpha = self.etas - self.etas_prev

        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = torch.cat((
            self.posterior_variance[1].unsqueeze(0),  # 将 self.posterior_variance[1] 扩展为 1 维
            self.posterior_variance[1:]  # 保持 self.posterior_variance[1:] 不变
                ))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance_clipped)


    def q_mean_variance(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = extract(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + extract(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute torche mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        # assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, x_t_xy_px, y, t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None
    ):
        
        if model_kwargs is None:
            model_kwargs = {}

        x_t = x_t_xy_px[:,0:1,:,:]

        B, C = x_t.shape[:2]
        assert t.shape == (B,)


        t0 = torch.zeros_like(t)

        f_hat = self.encoder(y, t0) # [b, 1, 32, 32]

        # 直接预测G
        pred_G = self.diff_model(x_t, t)
        # pred_G = y_0 - pred_res
        pred_u_hat = torch.sum(pred_G * f_hat, dim=1, keepdim=True).view(y.shape[0], 1, y.shape[2], y.shape[3])
        model_output = self.decoder(pred_u_hat, t0)   # 此时为x0不是res


        # model_output = self.model(x_t_xy_px, t)

        # model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = extract(self.posterior_variance, t, x_t.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        # pred_xstart = process_xstart(
        #         self._predict_xstart_from_residual(y=y, residual=model_output)
        #         )
        pred_xstart = process_xstart(model_output)

        # 这里输入的x_t是去噪的px
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        # assert (
        #     model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - extract(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - extract(self.etas, t, x_t.shape) * y
        ) / extract(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - extract(self.etas, t, x_t.shape) * y
        ) / extract(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - extract(1 - self.etas, t, x_t.shape) * pred_xstart
                - extract(self.etas, t, x_t.shape) * y
        ) / extract(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, x_xy_px, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):

        out = self.p_mean_variance(x_xy_px, y, t)
        x = x_xy_px[:,0:1,:,:]

        noise = torch.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        # torch.exp(0.5 * out["log_variance"]) = variance, 这么做的原因是为了防止操作特别大或者特别小的值
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}


    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor([self.T-1,] * y.shape[0], device=y.device).long()

        return y + extract(self.kappa * self.sqrt_etas, t, y.shape) * noise


    def p_sample_loop_progressive(
            self, u0, 
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):

        if device is None:
            device = "cuda:0"
        
        z_y = u0

        # generating noise
        if noise is None:
            noise = torch.randn_like(z_y)
            # noise = torch.zeros_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        z_sample_set = []


      
        # z_sample_xy =  torch.concat([z_sample, xy, a], dim=1)
        
        indices = list(range(self.T))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * z_sample.shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    z_sample,
                    z_y,
                    t
                )
                # yield out
                z_sample = out["sample"]
                # z_sample_xy =  torch.concat([z_sample, xy, a], dim=1)
            z_sample_set.append(z_sample.detach().cpu().squeeze().numpy())


        # np.save("DiffNo_CH", np.asarray(z_sample_set))
        return z_sample


    # G_0对应高清图像，px对应低清
    def forward(self, u0, u):
        
        pred_u = self.p_sample_loop_progressive(u0)

        err = torch.norm(pred_u - u, p=2) / torch.norm(u, p=2)

        batch_size = pred_u.shape[0]
        sample_errors = torch.zeros(batch_size)  # 用于存储每个样本的误差

        for i in range(batch_size):
            sample_errors[i] = torch.norm(pred_u[i] - u[i], p=2) / torch.norm(u[i], p=2)
        err = torch.mean(sample_errors)

        # loss = F.mse_loss(u, u_pred.squeeze(), reduction='mean')

        return pred_u, err


