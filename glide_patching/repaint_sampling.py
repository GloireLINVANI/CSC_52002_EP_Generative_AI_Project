import torch as th

class BaseGlideSampler:
    def __init__(self, model, diffusion, options, model_fn, device='cpu'):
        self.model = model
        self.diffusion = diffusion
        self.options = options
        self.device = device
        self.model_fn = model_fn
        if model_fn is None:
            self.model_fn = model

    def generate_model_kwargs(self, prompt, batch_size, batch_prompts=False):
        all_tokens = []
        all_mask = []

        if not batch_prompts:
            tokens = self.model.tokenizer.encode(prompt)
            tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
                tokens, self.options['text_ctx']
            )

            all_tokens = [tokens] * batch_size
            all_mask = [mask] * batch_size
        else:
            print("Batch prompts")
            for p in prompt:
                tokens = self.model.tokenizer.encode(p)
                tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
                    tokens, self.options['text_ctx']
                )

                all_tokens.append(tokens)
                all_mask.append(mask)

        model_kwargs = dict(
            tokens=th.tensor(
                all_tokens, device=self.device
            ),
            mask=th.tensor(
                all_mask,
                dtype=th.bool,
                device=self.device,
            ),
        )

        return model_kwargs
    
    def sample(self, prompt, batch_size, model_kwargs=None, batch_prompts=False, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs={}
        m_kwargs = self.generate_model_kwargs(prompt, batch_size, batch_prompts=batch_prompts)
        m_kwargs.update(model_kwargs)
        model_kwargs = m_kwargs

        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            self.model_fn,
            (batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            # cond_fn=None,
            **p_sample_kwargs
        )
        self.model.del_cache()
        return samples

class CFGSampler(BaseGlideSampler):
    def __init__(self, model, diffusion, options, guidance_scale, device='cpu'):
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)
        super().__init__(model, diffusion, options, model_fn, device)

    def generate_model_kwargs(self, prompt, batch_size, batch_prompts=False):
        all_tokens = []
        all_mask = []

        if not batch_prompts:
            tokens = self.model.tokenizer.encode(prompt)
            tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
                tokens, self.options['text_ctx']
            )

            all_tokens = [tokens] * batch_size
            all_mask = [mask] * batch_size
        else:
            for p in prompt:
                tokens = self.model.tokenizer.encode(p)
                tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
                    tokens, self.options['text_ctx']
                )
                
                all_tokens.append(tokens)
                all_mask.append(mask)

        # Create the classifier-free guidance tokens (empty)
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                all_tokens + [uncond_tokens] * batch_size, device=self.device
            ),
            mask=th.tensor(
                all_mask + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=self.device,
            ),
        )
        return model_kwargs
    
    def sample(self, prompt, batch_size, model_kwargs=None, batch_prompts=False, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs={}
        print("CFGSampler")
        full_batch_size = batch_size*2
        m_kwargs = self.generate_model_kwargs(prompt, batch_size, batch_prompts=batch_prompts)
        model_kwargs.update(m_kwargs)
        return super().sample(prompt, full_batch_size, model_kwargs, batch_prompts=batch_prompts, **p_sample_kwargs)
    
class UpscaleSampler(BaseGlideSampler):
    def sample(self, samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs):
        addl_kwargs = dict(
            low_res=((samples+1)*127.5).round()/127.5 - 1,
        )
        model_kwargs.update(addl_kwargs)
        up_shape = (batch_size, 3, self.options["image_size"], self.options["image_size"])
        p_sample_kwargs.update(dict(
            noise=th.randn(up_shape, device=self.device) * upsample_temp,
        ))
        return super().sample(prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class CFGSamplerInpaint(CFGSampler):
    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        
        if source_image_64.shape[0] != batch_size:
            source_image_64 = source_image_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_64 = source_image_64.to(self.device)
        
        if source_mask_64.shape[0] != batch_size:
            source_mask_64 = source_mask_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_64 = source_mask_64.to(self.device)

        addl_kwargs = dict(
                inpaint_image=(source_image_64 * source_mask_64).repeat(2, 1, 1, 1).to(self.device),
                inpaint_mask=source_mask_64.repeat(2, 1, 1, 1).to(self.device),
        )
        model_kwargs.update(addl_kwargs)

        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )
        
        p_sample_kwargs.update(dict(
            denoised_fn=denoised_fn
        ))

        return super().sample(prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class UpscaleSamplerInpaint(UpscaleSampler):
    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}

        if source_image_256.shape[0] != batch_size:
            source_image_256 = source_image_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_256 = source_image_256.to(self.device)
        
        if source_mask_256.shape[0] != batch_size:
            source_mask_256 = source_mask_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_256 = source_mask_256.to(self.device)

        addl_kwargs = dict(
                inpaint_image=(source_image_256 * source_mask_256),
                inpaint_mask=source_mask_256,
        )
        model_kwargs.update(addl_kwargs)
        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )
        p_sample_kwargs.update(dict(
            denoised_fn=denoised_fn
        ))
        return super().sample(samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class CFGSamplerRepaint(CFGSampler):
    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        
        if source_image_64.shape[0] != batch_size:
            source_image_64 = source_image_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_64 = source_image_64.to(self.device)
        
        if source_mask_64.shape[0] != batch_size:
            source_mask_64 = source_mask_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_64 = source_mask_64.to(self.device)
        
        addl_kwargs = dict(
            gt=(source_image_64).repeat(2, 1, 1, 1),
            gt_keep_mask=source_mask_64.repeat(2, 1, 1, 1),
        )

        model_kwargs.update(addl_kwargs)
        return super().sample(prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class UpscaleSamplerRepaint(UpscaleSampler):
    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}

        if source_image_256.shape[0] != batch_size:
            source_image_256 = source_image_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_256 = source_image_256.to(self.device)
        
        if source_mask_256.shape[0] != batch_size:
            source_mask_256 = source_mask_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_256 = source_mask_256.to(self.device)

        addl_kwargs = dict(
                gt=(source_image_256),
                gt_keep_mask=source_mask_256,
        )
        model_kwargs.update(addl_kwargs)
        return super().sample(samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class CFGSamplerRepaintInpaint(CFGSampler):
    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}

        if source_image_64.shape[0] != batch_size:
            source_image_64 = source_image_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_64 = source_image_64.to(self.device)
        
        if source_mask_64.shape[0] != batch_size:
            source_mask_64 = source_mask_64.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_64 = source_mask_64.to(self.device)

        addl_kwargs = dict(
                gt=(source_image_64).repeat(2, 1, 1, 1).to(self.device),
                gt_keep_mask=source_mask_64.repeat(2, 1, 1, 1).to(self.device),
                inpaint_image=(source_image_64 * source_mask_64).repeat(2, 1, 1, 1).to(self.device),
                inpaint_mask=source_mask_64.repeat(2, 1, 1, 1).to(self.device),
        )
        
        model_kwargs.update(addl_kwargs)
        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )
        p_sample_kwargs.update(dict(
            denoised_fn=denoised_fn
        ))
        return super().sample(prompt, batch_size, model_kwargs, **p_sample_kwargs)
    
class UpscaleSamplerRepaintInpaint(UpscaleSampler):
    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        
        if source_image_256.shape[0] != batch_size:
            source_image_256 = source_image_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_image_256 = source_image_256.to(self.device)
        
        if source_mask_256.shape[0] != batch_size:
            source_mask_256 = source_mask_256.repeat(batch_size, 1, 1, 1).to(self.device)
        else:
            source_mask_256 = source_mask_256.to(self.device)

        addl_kwargs = dict(
                gt=source_image_256,
                gt_keep_mask=source_mask_256,
                inpaint_image=(source_image_256 * source_mask_256),
                inpaint_mask=source_mask_256,
        )
        model_kwargs.update(addl_kwargs)
        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )
        p_sample_kwargs.update(dict(
            denoised_fn=denoised_fn
        ))
        return super().sample(samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs)