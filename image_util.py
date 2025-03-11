from PIL import Image
from typing import Tuple
import numpy as np
import torch as torch
import torch.nn.functional as F
from IPython.display import display

def read_image(path: str, size: int = 256) -> torch.Tensor:
    """Process the input image to the format expected by the model."""

    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


def read_mask(path: str, size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return (torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 255).int()[:, :1, :, :]

def show_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))


def process_image(input_image, size=256, device='cpu'):
    """Process the input image to the format expected by the model."""
    # Resize to 256x256
    img = Image.fromarray(input_image).resize((size, size), Image.LANCZOS)
    # Convert to tensor in the range [-1, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1
    # Rearrange dimensions to [batch, channels, height, width]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)


def process_mask(mask_image, size=256, device='cpu'):
    """Process the mask to the format expected by the model."""
    # Resize to 256x256
    mask = Image.fromarray(mask_image).convert('L').resize((size, size), Image.NEAREST)
    # Convert to tensor (0 for inpaint areas, 1 for keep areas)
    mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
    # Add dimensions for batch and channel, expand to 3 channels to match image
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    return mask_tensor.to(device)