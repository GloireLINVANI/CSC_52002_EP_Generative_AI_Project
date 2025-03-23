import torch as th
import torch._dynamo.config
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os

import repaint_sampling as RS
import repaint_patcher as RP
import prepare_glide_inpaint as PGI
from image_util import *

size = 64
large_size = 256

common_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(large_size),
    transforms.CenterCrop(large_size),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
with open('data/datasets/imagenet.json') as f:
    imagenet_labels = json.load(f)

def imagenet_target_transform(target):
    return imagenet_labels[int(datasets['imagenet_val'].classes[target])]



class CocoFolder(Dataset):
    def __init__(self, root, annotations_json, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        with open(annotations_json) as f:
            coco_json = json.load(f)

        annotations_by_id = {_ann['image_id']: _ann for _ann in coco_json['annotations']}
        coco_filename_to_annotations = {_img['file_name']: annotations_by_id[_img['id']]['caption'] for _img in coco_json['images']}
        self.classes = list(set(coco_filename_to_annotations.keys()))
        self.classes.sort()

        all_files = os.listdir(root)
        all_images = [f for f in all_files if f.endswith('.jpg')]
        self.image_paths = [os.path.basename(f) for f in all_images]
        self.targets = [coco_filename_to_annotations[os.path.basename(f)] for f in all_images]
        self.image_paths = [os.path.join(root, f) for f in self.image_paths]
        self.samples = list(zip(self.image_paths, self.targets))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

datasets = {
    'imagenet_val': ImageFolder('data/datasets/ILSVRC2012_img_val_subset', transform=common_transform, target_transform=imagenet_target_transform),
    'coco_val2017': CocoFolder('data/datasets/val2017', 'data/annotations/captions_val2017.json', transform=common_transform),
    'places_365_train': ImageFolder('data/datasets/places365_standard/train', transform=common_transform, target_transform=lambda x: p365t_classes[x].replace("_", " ")),
    'places_365_val': ImageFolder('data/datasets/places365_standard/val', transform=common_transform,  target_transform=lambda x: p365v_classes[x].replace("_", " ")),
}

p365t_classes = datasets['places_365_train'].classes
p365v_classes = datasets['places_365_val'].classes

masks = {
    # 'ex64': read_mask('data/masks/64/ex64.png', size=64),
    # 'genhalf': read_mask('data/masks/64/genhalf.png',size=64),
    # 'sr64': read_mask('data/masks/64/sr64.png',size=64),
    'thick': read_mask('data/masks/64/thick.png',size=64),
    'thin': read_mask('data/masks/64/thin.png',size=64),
    'vs64': read_mask('data/masks/64/vs64.png',size=64),
}

dataloaders = {name: DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True) for name, dataset in datasets.items()}

def main():
    cap = 160
    guidance_scale = 5.0

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda:2')

    model, diffusion, options = PGI.create_glide_generative(device=device, cuda=has_cuda, timesteps='250')
    model_nip, diffusion_nip, options_nip = PGI.create_glide_generative(device=device, cuda=has_cuda, timesteps='250', use_inpaint=False)


    from copy import deepcopy

    diffusion_rp = deepcopy(diffusion)
    diffusion_rp_nip = deepcopy(diffusion_nip)

    RP.patch_model_for_repaint(diffusion_rp)
    RP.patch_model_for_repaint(diffusion_rp_nip)

    base_sampler = RS.CFGSamplerInpaint(model, diffusion, options, guidance_scale, device=device)
    base_sampler_rp = RS.CFGSamplerRepaint(model_nip, diffusion_rp_nip, options_nip, guidance_scale, device=device)
    base_sampler_rpip = RS.CFGSamplerRepaintInpaint(model, diffusion_rp, options, guidance_scale, device=device)

    torch._dynamo.config.cache_size_limit = 128
    
    base_sampler.sample = torch.compile(base_sampler.sample, mode="reduce-overhead")
    base_sampler_rp.sample = torch.compile(base_sampler_rp.sample, mode="reduce-overhead")
    base_sampler_rpip.sample = torch.compile(base_sampler_rpip.sample, mode="reduce-overhead")


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

    for mask_name, mask in masks.items():
        print(f"Processing {mask_name}")
        for dataloader in dataloaders.items():
            name, dataloader = dataloader
            print(f"Processing {name}") 
            processed = 0
            batch_size = dataloader.batch_size
            batch_num = 0
            for source_image_64, prompts in dataloader:
                save_path_base = f"data/samples/{name}/{mask_name}/base/"
                save_path_rp = f"data/samples/{name}/{mask_name}/repaint/"
                save_path_rpip = f"data/samples/{name}/{mask_name}/rpip/"
                save_path_original = f"data/samples/{name}/{mask_name}/original/"
                os.makedirs(save_path_base, exist_ok=True)
                os.makedirs(save_path_rp, exist_ok=True)
                os.makedirs(save_path_rpip, exist_ok=True)
                os.makedirs(save_path_original, exist_ok=True)

                samples_glide = base_sampler.sample(source_image_64, mask, prompts, batch_size, batch_prompts=True)
                samples_glide = samples_glide[:batch_size]
                samples_repaint = base_sampler_rp.sample(source_image_64, mask, prompts, batch_size, jump_params=jump_params_rp_nip, batch_prompts=True)
                samples_repaint = samples_repaint[:batch_size]
                samples_rpip = base_sampler_rpip.sample(source_image_64, mask, prompts, batch_size, jump_params=jump_params, batch_prompts=True)
                samples_rpip = samples_rpip[:batch_size]

                save_batch(source_image_64, save_path_original + f"{batch_num}_" + '{1}_{0}.png', prompts)
                save_batch(samples_glide, save_path_base + f"{batch_num}_" + '{1}_{0}.png', prompts)
                save_batch(samples_repaint, save_path_rp + f"{batch_num}_" + '{1}_{0}.png', prompts)
                save_batch(samples_rpip, save_path_rpip + f"{batch_num}_" + '{1}_{0}.png', prompts)

                batch_num += 1
                processed += batch_size
                if processed > cap:
                    break
            print(f"Processed {processed} images")


if __name__ == '__main__':
    main()