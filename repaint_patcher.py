from collections import defaultdict
from tqdm.auto import tqdm
import torch


def get_schedule_jump(
    t_T,
    n_sample,
    jump_length,
    jump_n_sample,
    jump2_length=1,
    jump2_n_sample=1,
    jump3_length=1,
    jump3_n_sample=1,
    start_resampling=100000000,
    end_resampling=-1
):
    """RePaint's sampling schedule with jumps."""

    def init_jumps(jump_length, jump_n_sample):
        return {j: jump_n_sample - 1 for j in range(0, t_T - jump_length, jump_length)}

    jumps = init_jumps(jump_length, jump_n_sample)
    jumps2 = init_jumps(jump2_length, jump2_n_sample)
    jumps3 = init_jumps(jump3_length, jump3_n_sample)
    # print(f"jumps: {jumps}")
    # print(f"jumps2: {jumps2}")
    # print(f"jumps3: {jumps3}")

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if t + 1 < t_T - 1 and t <= start_resampling and t > end_resampling:
            ts.extend([t + 1, t] * n_sample)

        if jumps3.get(t, 0) > 0 and t <= start_resampling - jump3_length and t > end_resampling:
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if jumps2.get(t, 0) > 0 and t <= start_resampling - jump2_length and t > end_resampling:
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = init_jumps(jump3_length, jump3_n_sample)

        if jumps.get(t, 0) > 0 and t <= start_resampling - jump_length and t > end_resampling:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = init_jumps(jump2_length, jump2_n_sample)
            jumps3 = init_jumps(jump3_length, jump3_n_sample)

    ts.append(-1)

    return ts


def repaint_p_sample_loop_progressive(
    diffusion_model,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    conf=None,
    jump_params=None,
):
    """
    Generate samples from the model and yield intermediate samples from
    each timestep of diffusion.

    Arguments are the same as p_sample_loop().
    Returns a generator over dicts, where each dict is the return value of
    p_sample().
    """
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        image_after_step = noise
    else:
        image_after_step = torch.randn(*shape, device=device)

    pred_xstart = None

    idx_wall = -1
    sample_idxs = defaultdict(lambda: 0)

    times = get_schedule_jump(**jump_params)
    print(f"times: {times}")
    print(f"len(times): {len(times)}")

    time_pairs = list(zip(times[:-1], times[1:]))
    if progress:
        time_pairs = tqdm(time_pairs)

    for t_last, t_cur in time_pairs:
        idx_wall += 1
        t_last_t = torch.tensor(
            [t_last] * shape[0], device=device  # pylint: disable=not-callable
        )

        if t_cur < t_last:  # reverse
            with torch.no_grad():
                image_before_step = image_after_step.clone()
                out = diffusion_model.p_sample(
                    model,
                    image_after_step,
                    t_last_t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    # conf=conf,
                    pred_xstart=pred_xstart,
                )
                image_after_step = out["sample"]
                pred_xstart = out["pred_xstart"]

                sample_idxs[t_cur] += 1

                yield out

        else:
            t_shift = 1

            image_before_step = image_after_step.clone()
            image_after_step = diffusion_model.undo(
                image_before_step,
                image_after_step,
                est_x_0=out["pred_xstart"],
                t=t_last_t + t_shift,
                debug=False,
            )
            pred_xstart = out["pred_xstart"]


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def repaint_p_sample(
    diffusion_model,
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
):
    """
    Sample x_{t-1} from the model at the given timestep.

    :param model: the model to sample from.
    :param x: the current tensor at x_{t-1}.
    :param t: the value of t, starting at 0 for the first diffusion step.
    :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param cond_fn: if not None, this is a gradient function that acts
                    similarly to the model.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict containing the following keys:
             - 'sample': a random sample from the model.
             - 'pred_xstart': a prediction of x_0.
    """
    noise = torch.randn_like(x)

    if pred_xstart is not None:
        gt_keep_mask = model_kwargs.get("gt_keep_mask")

        gt = model_kwargs["gt"]

        alpha_cumprod = extract_into_tensor(diffusion_model.alphas_cumprod, t, x.shape)

        gt_weight = torch.sqrt(alpha_cumprod)
        gt_part = gt_weight * gt

        noise_weight = torch.sqrt((1 - alpha_cumprod))
        noise_part = noise_weight * torch.randn_like(x)

        weighed_gt = gt_part + noise_part

        x = gt_keep_mask * (weighed_gt) + (1 - gt_keep_mask) * (x)

    model_kwargs = {
        k: model_kwargs[k] for k in model_kwargs if k != "gt" and k != "gt_keep_mask"
    }

    out = diffusion_model.p_mean_variance(
        model,
        x,
        t,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )

    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

    if cond_fn is not None:
        out["mean"] = diffusion_model.condition_mean(
            cond_fn, out, x, t, model_kwargs=model_kwargs
        )

    sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    result = {
        "sample": sample,
        "pred_xstart": out["pred_xstart"],
        "gt": model_kwargs.get("gt"),
    }

    return result


def rp_undo(
    diffusion_model, image_before_step, img_after_model, est_x_0, t, debug=False
):
    beta = extract_into_tensor(diffusion_model.betas, t, img_after_model.shape)
    img_in_est = torch.sqrt(1 - beta) * img_after_model + torch.sqrt(
        beta
    ) * torch.randn_like(img_after_model)
    return img_in_est


def repaint_p_sample_loop(
    diffusion_model,
    model,
    shape,
    device=None,
    noise=None,
    clip_denoised=True,
    model_kwargs=None,
    progress=False,
    jump_params=None,
    **kwargs
):
    """Full sampling loop that returns final image."""
    result = repaint_p_sample_loop_progressive(
        diffusion_model=diffusion_model,
        model=model,
        shape=shape,
        noise=noise,
        device=device,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        progress=progress,
        jump_params=jump_params,
        **kwargs
    )
    final = None
    for sample in result:
        final = sample
    return final["sample"]


def patch_model_for_repaint(diffusion):
    """Set up RePaint sampling for a given diffusion model."""
    diffusion.p_sample = lambda *args, **kwargs: repaint_p_sample(
        diffusion, *args, **kwargs
    )
    diffusion.undo = lambda *args, **kwargs: rp_undo(diffusion, *args, **kwargs)
    diffusion.p_sample_loop = lambda *args, **kwargs: repaint_p_sample_loop(
        diffusion, *args, **kwargs
    )
