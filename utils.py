import os
import yaml

import torch
from torch.utils.data import DataLoader

from data import MRImageDataset_2D, MRImageDataset_3D

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory, exist_ok=True)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory, exist_ok=True)
    return checkpoint_directory, image_directory


def get_mr_data_loader(data, img_path, train, batch_size, dataset_name='UII', num_workers=1):
    if data == '2D':
        dataset = MRImageDataset_2D(img_path, dataset_name)
    elif data == '3D':
        dataset = MRImageDataset_3D(img_path, dataset_name)
    else:
        raise NotImplementedError

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train,
                        drop_last=train,
                        num_workers=num_workers)
    return loader


def k_loss(pred, gt):
    mag = torch.unsqueeze(abs(pred[..., 0].detach() + 1j * pred[..., 1].detach()), -1)
    loss = torch.mean(torch.div((pred - gt) ** 2, mag ** 2 + 1e-3))
    return loss


