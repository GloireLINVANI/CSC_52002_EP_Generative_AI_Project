import torch as th

class BaseGlideSampler:
    """
    Base class for Glide Sampler.
    Attributes:
        model: The model used for sampling.
        diffusion: The diffusion process used for sampling.
        options: Dictionary containing various options for the sampler.
        device: The device to run the model on (default is 'cpu').
        model_fn: The function to use for the model. If None, defaults to the model itself.
    """

    def __init__(self, model, diffusion, options, model_fn, device='cpu'):
        """
        Initializes the BaseGlideSampler with the given model, diffusion process, options, and device.
        Args:
            model: The model to be used for sampling.
            diffusion: The diffusion process to be used for sampling.
            options: Dictionary containing various options for the sampler.
            model_fn: The function to use for the model. If None, defaults to the model itself.
            device: The device to run the model on (default is 'cpu').
        """
        self.model = model
        self.diffusion = diffusion
        self.options = options
        self.device = device
        self.model_fn = model_fn
        if model_fn is None:
            self.model_fn = model

    def generate_model_kwargs(self, prompt, batch_size, batch_prompts=False):
        """
        Generates model keyword arguments based on the given prompt and batch size.
        Args:
            prompt: The text prompt to generate tokens from.
            batch_size: The number of samples to generate.
            batch_prompts: Whether to treat the prompt as a batch of prompts (default is False).
        Returns:
            A dictionary containing the tokens and mask tensors.
        """
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
        """
        Samples images based on the given prompt and batch size.
        Args:
            prompt: The text prompt to generate samples from.
            batch_size: The number of samples to generate.
            model_kwargs: Additional keyword arguments for the model (default is None).
            batch_prompts: Whether to treat the prompt as a batch of prompts (default is False).
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            A tensor containing the generated samples.
        """
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
    """
    CFGSampler is a subclass of BaseGlideSampler that implements classifier-free guidance sampling.
    Args:
        model (nn.Module): The model to be used for sampling.
        diffusion (Diffusion): The diffusion process to be used.
        options (dict): Additional options for the sampler.
        guidance_scale (float): The scale for classifier-free guidance.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
    Methods:
        generate_model_kwargs(prompt, batch_size, batch_prompts=False):
            Generates the model keyword arguments required for sampling.
            Args:
                prompt (str or list of str): The input prompt(s) for the model.
                batch_size (int): The number of samples to generate.
                batch_prompts (bool, optional): Whether to treat the prompt as a batch of prompts. Defaults to False.
            Returns:
                dict: A dictionary containing the tokens and mask for the model.
        sample(prompt, batch_size, model_kwargs=None, batch_prompts=False, **p_sample_kwargs):
            Samples from the model using the provided prompt and batch size.
            Args:
                prompt (str or list of str): The input prompt(s) for the model.
                batch_size (int): The number of samples to generate.
                model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
                batch_prompts (bool, optional): Whether to treat the prompt as a batch of prompts. Defaults to False.
                **p_sample_kwargs: Additional keyword arguments for the sampling process.
            Returns:
                Tensor: The generated samples.
    """

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
        """            
        Generates the model keyword arguments required for sampling.
        Args:
            prompt (str or list of str): The input prompt(s) for the model.
            batch_size (int): The number of samples to generate.
            batch_prompts (bool, optional): Whether to treat the prompt as a batch of prompts. Defaults to False.
        Returns:
            dict: A dictionary containing the tokens and mask for the model.
        """
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
        """
        Samples from the model using the provided prompt and batch size.
        Args:
            prompt (str or list of str): The input prompt(s) for the model.
            batch_size (int): The number of samples to generate.
            model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
            batch_prompts (bool, optional): Whether to treat the prompt as a batch of prompts. Defaults to False.
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            Tensor: The generated samples.
        """
        if model_kwargs is None:
            model_kwargs={}
        print("CFGSampler")
        full_batch_size = batch_size*2
        m_kwargs = self.generate_model_kwargs(prompt, batch_size, batch_prompts=batch_prompts)
        model_kwargs.update(m_kwargs)
        return super().sample(prompt, full_batch_size, model_kwargs, batch_prompts=batch_prompts, **p_sample_kwargs)
    
class UpscaleSampler(BaseGlideSampler):
    """
    UpscaleSampler is a class that inherits from BaseGlideSampler and is used to perform upsampling on given samples.
    Methods:
        sample(samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs)
            Performs upsampling on the provided samples using the specified parameters.
            Args:
                samples (tensor): The input samples to be upsampled.
                upsample_temp (float): The temperature parameter for the upsampling noise.
                prompt (str): The prompt to guide the upsampling process.
                batch_size (int): The number of samples in a batch.
                model_kwargs (dict, optional): Additional keyword arguments for the model.
                **p_sample_kwargs: Additional keyword arguments for the sampling process.
            Returns:
                tensor: The upsampled samples.
    """
    
    def sample(self, samples, upsample_temp, prompt, batch_size, model_kwargs, **p_sample_kwargs):
        '''
        Performs upsampling on the provided samples using the specified parameters.
        Args:
            samples (tensor): The input samples to be upsampled.
            upsample_temp (float): The temperature parameter for the upsampling noise.
            prompt (str): The prompt to guide the upsampling process.
            batch_size (int): The number of samples in a batch.
            model_kwargs (dict, optional): Additional keyword arguments for the model.
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            tensor: The upsampled samples.
        '''
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
    """
    CFGSamplerInpaint is a subclass of CFGSampler that provides functionality for inpainting images.
    Args:
        model (nn.Module): The model to be used for sampling.
        diffusion (Diffusion): The diffusion process to be used.
        options (dict): Additional options for the sampler.
        guidance_scale (float): The scale for classifier-free guidance.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
    Methods:
        sample(source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
            Samples images based on the provided source image, mask, and prompt.
            Args:
                source_image_64 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
                source_mask_64 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
                prompt (str): The text prompt to guide the image generation.
                batch_size (int): The number of images to generate in a batch.
                model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to None.
                **p_sample_kwargs: Additional keyword arguments to pass to the sampling function.
            Returns:
                torch.Tensor: The generated images tensor.
    """

    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Samples images based on the provided source image, mask, and prompt.
        Args:
            source_image_64 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
            source_mask_64 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
            prompt (str): The text prompt to guide the image generation.
            batch_size (int): The number of images to generate in a batch.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to None.
            **p_sample_kwargs: Additional keyword arguments to pass to the sampling function.
        Returns:
            torch.Tensor: The generated images tensor.
        """
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
    """
    UpscaleSamplerInpaint is a subclass of UpscaleSampler that provides functionality for inpainting during the upscaling process.
    Methods:
        sample(samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
            Samples the upscaled image with inpainting applied.
            Args:
                samples (torch.Tensor): The input samples to be upscaled.
                upsample_temp (float): The temperature parameter for the upsampling process.
                source_image_256 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
                source_mask_256 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
                prompt (str): The text prompt guiding the generation process.
                batch_size (int): The batch size for the sampling process.
                model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
                **p_sample_kwargs: Additional keyword arguments for the sampling process.
            Returns:
                torch.Tensor: The upscaled and inpainted image tensor.
    """

    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Samples the upscaled image with inpainting applied.
        Args:
            samples (torch.Tensor): The input samples to be upscaled.
            upsample_temp (float): The temperature parameter for the upsampling process.
            source_image_256 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
            source_mask_256 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
            prompt (str): The text prompt guiding the generation process.
            batch_size (int): The batch size for the sampling process.
            model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            torch.Tensor: The upscaled and inpainted image tensor.
        """
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
    """
    CFGSamplerRepaint is a subclass of CFGSampler that provides functionality for inpainting images, using RePaint sampling strategy.
    Args:
        model (nn.Module): The model to be used for sampling.
        diffusion (Diffusion): The diffusion process to be used.
        options (dict): Additional options for the sampler.
        guidance_scale (float): The scale for classifier-free guidance.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
    Methods:
        sample(source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
            Samples images based on the provided source image, mask, and prompt. Requires a jump_params kwarg.
            Args:
                source_image_64 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
                source_mask_64 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
                prompt (str): The text prompt to guide the image generation.
                batch_size (int): The number of images to generate in a batch.
                model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to None.
                **p_sample_kwargs: Additional keyword arguments to pass to the sampling function.
            Returns:
                torch.Tensor: The generated images tensor.
    """
    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Samples images based on the provided source image, mask, and prompt. Requires a jump_params kwarg.
        Args:
            source_image_64 (torch.Tensor): The source image tensor with shape (batch_size, channels, height, width).
            source_mask_64 (torch.Tensor): The source mask tensor with shape (batch_size, channels, height, width).
            prompt (str): The text prompt to guide the image generation.
            batch_size (int): The number of images to generate in a batch.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to None.
            **p_sample_kwargs: Additional keyword arguments to pass to the sampling function.
        Returns:
            torch.Tensor: The generated images tensor.
        """
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
    """
    UpscaleSamplerRepaint is a subclass of UpscaleSampler that provides functionality
    to sample images with additional parameters for source images and masks.
    Methods:
        sample(samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
            Samples images with the given parameters, including source images and masks. Requires a jump_params kwarg.
    """
    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Samples images with the given parameters, including source images and masks. Requires a jump_params kwarg.
        Args:
            samples (Tensor): The input samples to be upscaled.
            upsample_temp (float): The temperature parameter for upsampling.
            source_image_256 (Tensor): The source image tensor with shape (batch_size, channels, height, width).
            source_mask_256 (Tensor): The source mask tensor with shape (batch_size, channels, height, width).
            prompt (str): The prompt for the model.
            batch_size (int): The batch size for sampling.
            model_kwargs (dict, optional): Additional keyword arguments for the model (default is None).
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            Tensor: The upsampled images.
        """
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
    """
    CFGSamplerRepaintInpaint is a class that extends CFGSampler to perform sampling with inpainting capabilities.
    It handles the generation of samples by incorporating source images and masks, and ensures that the model
    predictions adhere to the known parts of the image.
    Methods:
        sample(source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
            Generates samples based on the provided source images, masks, and prompt. Requires a jump_params kwarg.
            Args:
                source_image_64 (Tensor): The source image tensor with shape (batch_size, channels, height, width).
                source_mask_64 (Tensor): The source mask tensor with shape (batch_size, channels, height, width).
                prompt (str): The text prompt to guide the sampling process.
                batch_size (int): The number of samples to generate in a batch.
                model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
                **p_sample_kwargs: Additional keyword arguments for the sampling process.
            Returns:
                Tensor: The generated samples.
    """
    def sample(self, source_image_64, source_mask_64, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Generates samples based on the provided source images, masks, and prompt. Requires a jump_params kwarg.
        Args:
            source_image_64 (Tensor): The source image tensor with shape (batch_size, channels, height, width).
            source_mask_64 (Tensor): The source mask tensor with shape (batch_size, channels, height, width).
            prompt (str): The text prompt to guide the sampling process.
            batch_size (int): The number of samples to generate in a batch.
            model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            Tensor: The generated samples.
        """
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
    """
    A class that extends the UpscaleSampler to perform inpainting during the upscaling process.
    """
    def sample(self, samples, upsample_temp, source_image_256, source_mask_256, prompt, batch_size, model_kwargs=None, **p_sample_kwargs):
        """
        Samples images with inpainting during the upscaling process. Requires a jump_params kwarg.
        Args:
            samples (Tensor): The input samples to be upscaled.
            upsample_temp (float): The temperature parameter for upsampling.
            source_image_256 (Tensor): The source image tensor of shape (batch_size, channels, height, width).
            source_mask_256 (Tensor): The source mask tensor of shape (batch_size, channels, height, width).
            prompt (str): The text prompt for the model.
            batch_size (int): The batch size for processing.
            model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
            **p_sample_kwargs: Additional keyword arguments for the sampling process.
        Returns:
            Tensor: The upscaled and inpainted samples.
        """
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