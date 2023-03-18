from typing import Tuple
import torch
import torchvision
from torchvision.io import read_image
import os
from pathlib import Path


def load_exampe(folder: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # load images
    target = read_image(os.path.join(folder, 'target_in.png')) / 255
    ctx_in = []
    ctx_out = []
    for i in range(32):
        if not Path(folder, f'ctx_in{i}.png').is_file():
            break
        ctx_in.append(read_image(os.path.join(folder, f'ctx_in{i}.png')) / 255)
        ctx_out.append(read_image(os.path.join(
            folder, f'ctx_out{i}.png')) / 255)
    ctx_in = torch.stack(ctx_in, dim=0)
    ctx_out = torch.stack(ctx_out, dim=0)

    # pad to 3 channels (we could use these channels to pass additional modalities, targets, etc ...)
    def pad_to_3_chan(t):
        pad = torch.zeros_like(t)
        return torch.concat([t, pad, pad], dim=-3)
    target = pad_to_3_chan(target)
    ctx_in = pad_to_3_chan(ctx_in)
    ctx_out = pad_to_3_chan(ctx_out)

    # combine and add batch dim
    target = target.unsqueeze(0)
    ctx_in = ctx_in.unsqueeze(0)
    ctx_out = ctx_out.unsqueeze(0)

    return target, ctx_in, ctx_out


def display_images(tensor: torch.Tensor):
    tensor = tensor.squeeze(0)
    if tensor.ndim > 3:
        tensor = torchvision.utils.make_grid(tensor[:, [0]], padding=0)
    else:
        tensor = tensor[[0]]
    display(torchvision.transforms.ToPILImage()(tensor))


def corp_exampe(folder: str):
    # load images
    target = read_image(os.path.join(folder, 'target_in.png'))
    target = target[0, :192, :192]
    torchvision.transforms.ToPILImage()(target).save(
        os.path.join(folder, 'target_in.png'))
    for i in range(32):
        if not Path(folder, f'ctx_in{i}.png').is_file():
            break

        target = read_image(os.path.join(folder, f'ctx_in{i}.png'))
        target = target[0, :192, :192]
        torchvision.transforms.ToPILImage()(target).save(
            os.path.join(folder, f'ctx_in{i}.png'))
        target = read_image(os.path.join(folder, f'ctx_out{i}.png'))
        target = target[0, :192, :192]
        torchvision.transforms.ToPILImage()(target).save(
            os.path.join(folder, f'ctx_out{i}.png'))
