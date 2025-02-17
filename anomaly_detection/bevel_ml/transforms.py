import cv2
import numpy as np
import torch


def tone_curve(img: torch.Tensor, alpha: int, beta: int):
    img_ary = img.squeeze(0).numpy().astype(dtype=np.uint8)
    lut = np.zeros((256), dtype="uint8")
    for i in range(256):
        val = (alpha * i) - beta
        val = 255 if val > 255 else val
        val = 0 if val < 0 else val
        lut[i] = val
    lut = lut.astype(np.uint8)
    res = cv2.LUT(img_ary, lut)
    return torch.from_numpy(res).unsqueeze(0)


def check_input(img: torch.Tensor):
    assert len(img.shape) == 3 or 4
    shape = img.shape
    if len(img.shape) == 4:
        img = img.squeeze(0)
    return img, shape


class ChangeContrast(torch.nn.Module):
    def __init__(self, apply: bool = False):
        self.apply = apply

    def __call__(self, img: torch.Tensor):
        img, shape = check_input(img)
        alpha1, beta1 = (3.6, 200)
        alpha2, beta2 = (3.0, 360)
        alpha3, beta3 = (3.6, 540)
        if self.apply:
            img[0] = tone_curve(img[0], alpha1, beta1)
            img[1] = tone_curve(img[1], alpha2, beta2)
            img[2] = tone_curve(img[2], alpha3, beta3)
        return img.reshape(shape)
