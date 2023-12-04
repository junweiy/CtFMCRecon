import torch
import torch.fft
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.colors as colors

def rss_complex(data):
    return np.sqrt((data ** 2).sum(-1).sum(0, keepdims=True))

def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho')
    return x




def torch_fft(tensor, dim=(-2, -1)):
    return fftshift(torch.fft.fftn(tensor, dim=dim), dim=dim)


def torch_ifft(tensor, dim=(-2, -1)):
    return torch.fft.ifftn(ifftshift(tensor, dim), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return torch.roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return torch.roll(x, shift, dim)


def get_ssim(pred, gt, domain):
    if torch.is_tensor(pred):
        if domain == 0:
            pred = k_to_i(pred)
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        if domain == 0:
            gt = k_to_i(gt)
        gt = gt.cpu().numpy()
    img_size = pred.shape
    # 2D (1, x, y, 2) or 3D (n/c, x, y, 2)
    if len(img_size) == 4:
        if img_size[0] == 1:
            return structural_similarity(pred[0], gt[0], channel_axis=-1)
        else:
            total_ssim = 0
            for sl in range(img_size[0]):
                total_ssim += structural_similarity(pred[sl], gt[sl], channel_axis=-1)
            return total_ssim / img_size[0]
    # 3D (1, z, x, y, 2)
    elif len(img_size) == 5:
        assert img_size[0] == 1
        total_ssim = 0
        for sl in range(img_size[1]):
            total_ssim += structural_similarity(pred[0,sl], gt[0,sl], channel_axis=-1)
        return total_ssim / img_size[1]
    elif len(img_size) == 3:
        return structural_similarity(pred, gt, channel_axis=-1)
    else:
        raise NotImplementedError

def get_psnr(pred, gt, domain):
    if torch.is_tensor(pred):
        if domain == 0:
            pred = k_to_i(pred)
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        if domain == 0:
            gt = k_to_i(gt)
        gt = gt.cpu().numpy()
    img_size = pred.shape
    # 2D (1, x, y, 2) or 3D (n/c, x, y, 2)
    if len(img_size) == 4:
        if img_size[0] == 1:
            return psnr(pred[0], gt[0])
        else:
            total_psnr = 0
            for sl in range(img_size[0]):
                total_psnr += psnr(pred[sl], gt[sl])
            return total_psnr / img_size[0]
    # 3D (1, z, x, y, 2)
    elif len(img_size) == 5:
        assert img_size[0] == 1
        total_psnr = 0
        for sl in range(img_size[1]):
            total_psnr += psnr(pred[0,sl], gt[0,sl])
        return total_psnr / img_size[1]
    elif len(img_size) == 3:
        return psnr(pred, gt)
    else:
        raise NotImplementedError

def k_to_i(tensor):
    if len(tensor.shape) == 3:
        iffted = torch_ifft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    elif len(tensor.shape) == 4:
        iffted = torch_ifft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    elif len(tensor.shape) == 5:
        iffted = torch_ifft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    else:
        NotImplementedError
    return torch.stack((iffted.real, iffted.imag), -1)

def i_to_k(tensor):
    if len(tensor.shape) == 3:
        ffted = torch_fft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    elif len(tensor.shape) == 4:
        ffted = torch_fft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    elif len(tensor.shape) == 5:
        ffted = torch_fft(tensor[..., 0] + 1j * tensor[..., 1], (-2, -1))
    else:
        NotImplementedError
    return torch.stack((ffted.real, ffted.imag), -1)


def psnr(x, y, max_val=1):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape

    max_intensity = max_val
    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def DC(pred, gt, mask, domain):
    if domain == 1:
        pred = i_to_k(pred)
        gt = i_to_k(gt)
    res = gt * mask + pred * (1 - mask)
    if domain == 1:
        res = k_to_i(res)
    return res.type(torch.FloatTensor)

