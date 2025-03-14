from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
def create_glide_generative(device, use_inpaint=True, cuda=False, timesteps='250'):
    options = model_and_diffusion_defaults()
    options['inpaint'] = use_inpaint
    options['use_fp16'] = cuda
    options['timestep_respacing'] = timesteps
    model, diffusion = create_model_and_diffusion(**options)

    if cuda:
        model.convert_to_fp16()
    model.to(device)
    if use_inpaint:
        model.load_state_dict(load_checkpoint('base-inpaint', device))
    else:
        model.load_state_dict(load_checkpoint('base', device))
    return model, diffusion, options

def create_glide_upsampler(device, use_inpaint=True, cuda=False, timesteps='250'):
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = use_inpaint
    options_up['use_fp16'] = cuda
    options_up['timestep_respacing'] = timesteps

    model, diffusion = create_model_and_diffusion(**options_up)
    model.eval()
    if cuda:
        model.convert_to_fp16()
    model.to(device)
    if use_inpaint:
        model.load_state_dict(load_checkpoint('upsample-inpaint', device))
    else:
        model.load_state_dict(load_checkpoint('upsample', device))
    return model, diffusion, options_up