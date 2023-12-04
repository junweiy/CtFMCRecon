import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=h),
                                     torch.linspace(-1, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=c), \
                                            torch.linspace(-1, 1, steps=h), \
                                            torch.linspace(-1, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


class MRImageDataset_2D(Dataset):

    def __init__(self, img_path, dataset_name):
        '''
        img_dim: new image size [h, w]
        '''
        h5_file = h5py.File(img_path, 'r')
        t1_k = h5_file['T1'][()]
        t2_k = h5_file['T2'][()]
        t2f_k = h5_file['FLAIR'][()]
        h5_file.close()

        t1_img = np.fft.ifft2(np.fft.ifftshift(t1_k))
        t2_img = np.fft.ifft2(np.fft.ifftshift(t2_k))
        t2f_img = np.fft.ifft2(np.fft.ifftshift(t2f_k))

        self.t1_k = torch.tensor(np.stack((t1_k.real, t1_k.imag), axis=-1))
        self.t2_k = torch.tensor(np.stack((t2_k.real, t2_k.imag), axis=-1))
        self.t2f_k = torch.tensor(np.stack((t2f_k.real, t2f_k.imag), axis=-1))

        self.img_dim = self.t1_k.shape[1:3]

        self.t1_img = torch.tensor(np.stack((t1_img.real, t1_img.imag), axis=-1)).type(torch.FloatTensor)
        self.t2_img = torch.tensor(np.stack((t2_img.real, t2_img.imag), axis=-1)).type(torch.FloatTensor)
        self.t2f_img = torch.tensor(np.stack((t2f_img.real, t2f_img.imag), axis=-1)).type(torch.FloatTensor)

    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, (self.t1_k[idx], self.t2_k[idx], self.t2f_k[idx]), (self.t1_img[idx], self.t2_img[idx], self.t2f_img[idx])

    def __len__(self):
        return 1


class MRImageDataset_3D(Dataset):

    def __init__(self, img_path, dataset_name='UII'):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.dataset_name = dataset_name
        h5_file = h5py.File(img_path, 'r')

        t1_k = h5_file['T1'][()]
        t2_k = h5_file['T2'][()]
        t2f_k = h5_file['FLAIR'][()]
        h5_file.close()

        self.img_dim = t1_k.shape[1:]

        t1_img = np.fft.ifft2(np.fft.ifftshift(t1_k, axes=(-2, -1)))
        t2_img = np.fft.ifft2(np.fft.ifftshift(t2_k, axes=(-2, -1)))
        t2f_img = np.fft.ifft2(np.fft.ifftshift(t2f_k, axes=(-2, -1)))

        self.t1_k = torch.tensor(np.stack((t1_k.real, t1_k.imag), axis=-1))[0]
        self.t2_k = torch.tensor(np.stack((t2_k.real, t2_k.imag), axis=-1))[0]
        self.t2f_k = torch.tensor(np.stack((t2f_k.real, t2f_k.imag), axis=-1))[0]

        self.t1_img = torch.tensor(np.stack((t1_img.real, t1_img.imag), axis=-1))[0]
        self.t2_img = torch.tensor(np.stack((t2_img.real, t2_img.imag), axis=-1))[0]
        self.t2f_img = torch.tensor(np.stack((t2f_img.real, t2f_img.imag), axis=-1))[0]


    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid[idx], (self.t1_k[idx], self.t2_k[idx], self.t2f_k[idx]), (self.t1_img[idx], self.t2_img[idx], self.t2f_img[idx])


    def __len__(self):
        return self.t1_k.shape[0]

