import os
from torch.utils.data import Dataset, DataLoader
import glide_patching.repaint_sampling as RS
import glide_patching.repaint_patcher as RP
import glide_patching.prepare_glide_inpaint as PGI
from glide_patching.image_util import *
import torch as th

import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        indices = list(range(len(self.targets)))
        indices.sort(key=self.targets.__getitem__)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], size=64)
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target
    
class CustomImageDatasetFolders(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        indices = list(range(len(self.targets)))
        indices.sort(key=self.targets.__getitem__)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.image_paths = [os.path.join(p, os.listdir(p)[0]) for p in self.image_paths]
        self.targets = [self.targets[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], size=256)
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

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
                for filename in os.listdir(model_path):
                    if filename.endswith('.png'):
                        prompt, id_ = filename[:-4].rsplit('_', 1)
                        prompt = prompt.replace('_', ' ')
                        image_paths.append(os.path.join(model_path, filename))
                        targets.append(prompt)

                datasets[dset][model] = CustomImageDataset(image_paths, targets, transform=transform)
            else:
                image_paths = []
                targets = []
                for filename in os.listdir(model_path, ):
                    if os.path.isdir(os.path.join(model_path, filename)):
                        prompt, id_ = filename.rsplit('_', 1)
                        prompt = prompt.replace('_', ' ')
                        image_paths.append(os.path.join(model_path, filename))
                        targets.append(prompt)

                datasets[dset][model] = CustomImageDatasetFolders(image_paths, targets, transform=transform)

    return datasets

if __name__ == "__main__":
    datasets = collect_images()
    datasets['places_365_train']['repaint'][2]

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda:2')

    model_up_nip, diffusion_up_nip, options_up_nip = PGI.create_glide_upsampler(device=device, cuda=has_cuda, timesteps='250', use_inpaint=False)

    guidance_scale = 7

    from copy import deepcopy

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

    upscaler_base = RS.UpscaleSampler(model_up_nip, diffusion_up_nip, options_up_nip, None, device=device)

    upscaler_base.sample = th.compile(upscaler_base.sample, mode="max-autotune")
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
            for i in range(batch_i, batch_i+batch_size):
                if i > len(dataset_d['original']):
                    break
                prompts.append(dataset_d['original'][i][1])
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
            up_base = upscaler_base.sample(base_image_64, 0.997, prompts, batch_size, {}, batch_prompts=True)
            up_base_rp = upscaler_base.sample(rp_image_64, 0.997, prompts, batch_size, {}, batch_prompts=True)
            up_base_rpip = upscaler_base.sample(rpip_image_64, 0.997, prompts, batch_size, {}, batch_prompts=True)
            os.makedirs(f'data/large_samples/{name}/base', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/repaint', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/rpip', exist_ok=True)
            os.makedirs(f'data/large_samples/{name}/original', exist_ok=True)
            save_batch(up_base, f'data/large_samples/{name}/base/' + '{0}_{1}.png', prompts)
            save_batch(up_base_rp, f'data/large_samples/{name}/repaint/' + '{0}_{1}.png', prompts)
            save_batch(up_base_rpip, f'data/large_samples/{name}/rpip/' + '{0}_{1}.png', prompts)
            save_batch(orig_image_256, f'data/large_samples/{name}/original/' + '{0}_{1}.png', prompts)
