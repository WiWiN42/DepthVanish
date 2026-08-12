import os
import tqdm
import torch as th
import numpy as np
from collections import defaultdict
from .gaussian_diffusion import _extract_into_tensor
from .respace import SpacedDiffusion
# from utils import normalize_image, save_grid
# from src.utils import pil_sample, get_grid_mask
# from utils.resize_right import resize

class DDNMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(use_timesteps, **kwargs)
        self.sigma_y = conf.get("DDNM.sigma_y", 0.0)
        self.eta = conf.get("DDNM.eta", 0.85)
        self.mode = conf.get("mode", "inpaint")
        # self.mode = conf.get("mode", "super_resolution")
        self.scale = conf.get("scale", 0)

    # Code form RePaint   
    def get_schedule_jump(self, T_sampling, travel_length, travel_repeat):
        jumps = {}
        for j in range(0, T_sampling - travel_length, travel_length):
            jumps[j] = travel_repeat - 1

        t = T_sampling
        ts = []

        while t >= 1:
            t = t-1
            ts.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(travel_length):
                    t = t + 1
                    ts.append(t)

        ts.append(-1)

        self._check_times(ts, -1, T_sampling)
        return ts

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = th.split(model_output, C, dim=1)
        return model_output

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        pred_xstart=None,
        idx_wall=-1,
        sample_dir=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if cond_fn is not None:
            model_fn = self._wrap_model(model)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
            with th.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient


        A = kwargs.get("A")
        Ap = kwargs.get("Ap")
        x0 = model_kwargs["img"]
        y = A(x0)
        # y = A(model_kwargs['sup'])
        mask = model_kwargs["mask"]

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        e_t = self._get_et(model, x, t, model_kwargs)
        alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        prev_t = t - 1
        alpha_prev = _extract_into_tensor(
            self.alphas_cumprod, prev_t, x.shape)
        sigma_t = (1 - alpha_t**2).sqrt()

        with th.no_grad():
            pred_x0 = process_xstart(
                (x - e_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt())

            single_sigma_t = sigma_t[0][0][0][0]
            single_alpha_t = alpha_prev[0][0][0][0]
            if single_sigma_t >= single_alpha_t * self.sigma_y:
                lambda_t = 1.0
                gamma_t = (sigma_t**2 - (alpha_prev * self.sigma_y) ** 2).sqrt()
            else:
                lambda_t = (sigma_t) / (alpha_prev * self.sigma_y)
                gamma_t = 0.0

            # DDNM modification
            pred_x0 = pred_x0 - lambda_t * Ap(A(pred_x0) - y)

            eta = self.eta
            c1 = (1 - alpha_prev).sqrt() * eta
            c2 = (1 - alpha_prev).sqrt() * ((1 - eta**2) ** 0.5)

            x_prev = alpha_prev.sqrt() * pred_x0 + gamma_t * (
                c1 * th.randn_like(pred_x0) + c2 * e_t
            )

        result = {
            "sample": x_prev,
            "pred_xstart": pred_x0,
            "gt": model_kwargs["img"],
        }
        return result

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        start_time_steps=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None,
        D=4,
        scale=2,
        save_prefix=None,
        **kwargs,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        mask = model_kwargs["mask"]
        def A(z): return z * (1-mask)
        Ap = A

        self.gt_noises = None
        pred_xstart = None
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        # if sample_dir is not None:
        #     os.makedirs(sample_dir, exist_ok=True)

        times = self.get_schedule_jump(
            T_sampling=start_time_steps,
            **conf.DDNM
        )

        time_pairs = list(zip(times[:-2], times[1:-1]))
        if progress:
            from tqdm.auto import tqdm

            time_pairs = tqdm(time_pairs)

        shape_u = (shape[0], 3, shape[2], shape[3])
        shape_d = (shape[0], 3, int(shape[2] / D), int(shape[3] / D))

        for t_last, t_cur in time_pairs:
            t_last_t = th.tensor([t_last] * shape[0], device=device)
            if t_cur < t_last:
                # denoise

                image_before_step = image_after_step.clone()
                image_after_step = image_after_step.requires_grad_()
                out = self.p_sample(
                    model,
                    image_after_step,
                    t_last_t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    conf=conf,
                    pred_xstart=pred_xstart,
                    A=A,
                    Ap=Ap,
                )
                # with th.enable_grad():
                #     align_x_part = out['pred_xstart'] * (1-model_kwargs['cp_mask'])
                #     align_xcp_part = model_kwargs['cp_img'] * (1-model_kwargs['cp_mask'])
                #     difference = \
                #     resize(
                #         resize(
                #             align_xcp_part,
                #             scale_factors=1.0/D,
                #             out_shape=shape_d,
                #         ),
                #         scale_factors=D,
                #         out_shape=shape_u,
                #     ) - \
                #     resize(
                #         resize(
                #             align_x_part,
                #             scale_factors=1.0/D,
                #             out_shape=shape_d,
                #         ),
                #         scale_factors=D,
                #         out_shape=shape_u,
                #     )
                #     norm = th.linalg.norm(difference)
                #     norm_grad = th.autograd.grad(outputs=norm, inputs=image_after_step)[0]
                #     out["sample"] -= norm_grad * scale

                yield out
                image_after_step = out["sample"]
                image_after_step = image_after_step.detach_()
                pred_xstart = out["pred_xstart"]

                # visualize middle diffusion results
                # pil_sample(
                #     [image_after_step],
                #     ['diff'],
                #     conf.TEST_SAMPLE_INTERVAL.index,
                #     os.path.join(conf.SAMPLE_DEST, f'{start_time_steps}diff'),
                #     shape=(1024, 768),
                #     name_prefix='{}_step_{}'.format(save_prefix, t_last),
                #     assemble_masked=False
                # )
            else:
                # ad dnoise
                t_shift = conf.get("inpa_inj_time_shift", 1)

                image_before_step = image_after_step.clone()
                image_after_step = self.undo(
                    image_before_step,
                    image_after_step,
                    est_x_0=out["pred_xstart"],
                    t=t_last_t + t_shift,
                    debug=False,
                )
                image_after_step = image_after_step.detach_()
                pred_xstart = out["pred_xstart"]
