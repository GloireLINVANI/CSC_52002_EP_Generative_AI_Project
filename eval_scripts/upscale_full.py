import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glide_patching.repaint_sampling as RS
import glide_patching.repaint_patcher as RP
import glide_patching.prepare_glide_inpaint as PGI
from glide_patching.image_util import *
import torch as th

import torchvision.transforms as transforms

masks_large = {
    'ex64': read_mask('data/masks/64/ex64.png', size=256),
    'genhalf': read_mask('data/masks/64/genhalf.png',size=256),
    'sr64': read_mask('data/masks/64/sr64.png',size=256, resample=Image.NEAREST),
    'thick': read_mask('data/masks/64/thick.png',size=256),
    'thin': read_mask('data/masks/64/thin.png',size=256),
    'vs64': read_mask('data/masks/64/vs64.png',size=256, resample=Image.NEAREST),
}

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, targets, mask_names, ids, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.mask_names = mask_names
        indices = list(range(len(self.targets)))
        indices.sort(key=ids.__getitem__)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.mask_names = [self.mask_names[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], size=64)
        target = self.targets[idx]
        mask_name = self.mask_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, (target, mask_name)
    
class CustomImageDatasetFolders(Dataset):
    def __init__(self, image_paths, targets, mask_names, ids, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.mask_names = mask_names
        indices = list(range(len(self.targets)))
        indices.sort(key=ids.__getitem__)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.image_paths = [os.path.join(p, os.listdir(p)[0]) for p in self.image_paths]
        self.targets = [self.targets[i] for i in indices]
        self.mask_names = [self.mask_names[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], size=256)
        target = self.targets[idx]
        mask_name = self.mask_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, (target, mask_name)

def collect_images(base_path='samples_to_upscale'):
    datasets = {}
    transform = None

    for dset in os.listdir(base_path):
        dset_path = os.path.join(base_path, dset)
        if not os.path.isdir(dset_path):
            continue

        datasets[dset] = {}
        for model in os.listdir(dset_path):
            model_path = os.path.join(dset_path, model)
            if not os.path.isdir(model_path):
                continue
            if model != 'originals':
                image_paths = []
                targets = []
                mask_names = []
                ids = []
                for filename in os.listdir(model_path):
                    if filename.endswith('.png'):
                        promptid, mask_name = filename[:-4].rsplit('_', 1)
                        prompt, id_ = promptid.rsplit('_', 1)
                        prompt = prompt.replace('_', ' ')
                        image_paths.append(os.path.join(model_path, filename))
                        targets.append(prompt)
                        mask_names.append(mask_name)
                        ids.append(filename[:-4])

                datasets[dset][model] = CustomImageDataset(image_paths, targets, mask_names, ids, transform=transform)
            else:
                image_paths = []
                targets = []
                mask_names = []
                ids = []
                for filename in os.listdir(model_path, ):
                    if os.path.isdir(os.path.join(model_path, filename)):
                        promptid, mask_name = filename.rsplit('_',1)
                        prompt, id_ = promptid.rsplit('_', 1)
                        prompt = prompt.replace('_', ' ')
                        image_paths.append(os.path.join(model_path, filename))
                        targets.append(prompt)
                        mask_names.append(mask_name)
                        ids.append(filename)

                datasets[dset][model] = CustomImageDatasetFolders(image_paths, targets, mask_names, ids, transform=transform)

    return datasets

if __name__ == "__main__":
    datasets = collect_images()

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda:1')
    model_up, diffusion_up, options_up = PGI.create_glide_upsampler(device=device, cuda=has_cuda, timesteps='250')
    model_up_nip, diffusion_up_nip, options_up_nip = PGI.create_glide_upsampler(device=device, cuda=has_cuda, timesteps='250', use_inpaint=False)

    guidance_scale = 7

    upscale_sampler = RS.UpscaleSamplerInpaint(model_up, diffusion_up, options_up, model_fn=None, device=device)

    from copy import deepcopy

    diffusion_rp_up = deepcopy(diffusion_up)

    RP.patch_model_for_repaint(diffusion_rp_up)

    diffusion_rp_up_nip = deepcopy(diffusion_up_nip)

    RP.patch_model_for_repaint(diffusion_rp_up_nip)

    jump_params = {
        "t_T": 250,
        "n_sample": 1,
        "jump_length": 10,
        "jump_n_sample": 6,
        "start_resampling": 30
    }

    jump_params_rp_nip = {
        "t_T": 250,
        "n_sample": 1,
        "jump_length": 10,
        "jump_n_sample": 5,
        "start_resampling": 150,
        "end_resampling": 50
    }
    upscale_sampler_rp = RS.UpscaleSamplerRepaint(model_up_nip, diffusion_rp_up_nip, options_up_nip, model_fn=None, device=device)

    upscale_sampler_rpip = RS.UpscaleSamplerRepaintInpaint(model_up, diffusion_rp_up, options_up, model_fn=None, device=device)

    upscale_sampler.sample = torch.compile(upscale_sampler.sample, mode="max-autotune")
    upscale_sampler_rp.sample = torch.compile(upscale_sampler_rp.sample, mode="max-autotune")
    upscale_sampler_rpip.sample = torch.compile(upscale_sampler_rpip.sample, mode="max-autotune")

    batch_size = 32

    res = {}
    for name, dataset_d in datasets.items():
        for batch_i in range(0,635,batch_size):
            orig_image_64 = []
            base_image_64 = []
            rp_image_64 = []
            rpip_image_64 = []
            orig_image_256 = []
            prompts = []
            masks = []
            for i in range(batch_i, batch_i+batch_size):
                if i > len(dataset_d['original']):
                    break
                prompts.append(dataset_d['original'][i][1][0])
                assert(dataset_d['original'][i][1][0] == dataset_d['base'][i][1][0] 
                        and dataset_d['original'][i][1][0] == dataset_d['repaint'][i][1][0]
                        and dataset_d['original'][i][1][0] == dataset_d['rpip'][i][1][0]
                        and dataset_d['original'][i][1][0] == dataset_d['originals'][i][1][0])
                masks.append(masks_large[dataset_d['original'][i][1][1]])
                orig_image_64.append(dataset_d['original'][i][0])
                base_image_64.append(dataset_d['base'][i][0])
                rp_image_64.append(dataset_d['repaint'][i][0])
                rpip_image_64.append(dataset_d['rpip'][i][0])
                orig_image_256.append(dataset_d['originals'][i][0])
            if(len(prompts) < batch_size):
                break
            orig_image_64 = torch.cat(orig_image_64).to(device)
            base_image_64 = torch.cat(base_image_64).to(device)
            rp_image_64 = torch.cat(rp_image_64).to(device)
            rpip_image_64 = torch.cat(rpip_image_64).to(device)
            orig_image_256 = torch.cat(orig_image_256).to(device)
            orig_mask_256 = torch.cat(masks).to(device)
            up_base = upscale_sampler.sample(base_image_64, 0.997, orig_image_256, orig_mask_256, prompts, batch_size, batch_prompts=True,)
            up_base_rp = upscale_sampler_rp.sample(rp_image_64, 0.997, orig_image_256, orig_mask_256, prompts, batch_size, batch_prompts=True,jump_params=jump_params_rp_nip)
            up_base_rpip = upscale_sampler_rpip.sample(rpip_image_64, 0.997, orig_image_256, orig_mask_256, prompts, batch_size, batch_prompts=True,jump_params=jump_params)
            os.makedirs(f'data/large_samples/{name}/base', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/repaint', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/rpip', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/original', exist_ok=True)
            save_batch(up_base, f'data/large_samples/{name}/base/' + '{0}_{1}.png', prompts)
            save_batch(up_base_rp, f'data/large_samples/{name}/repaint/' + '{0}_{1}.png', prompts)
            save_batch(up_base_rpip, f'data/large_samples/{name}/rpip/' + '{0}_{1}.png', prompts)
            save_batch(orig_image_256, f'data/large_samples/{name}/original/' + '{0}_{1}.png', prompts)
