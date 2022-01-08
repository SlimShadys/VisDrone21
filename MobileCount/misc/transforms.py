import numpy as np
from PIL import Image
import torch

class RandomHorizontallyFlip(object):
    def __call__(self, img, gt, bbx=None):
        if not(issubclass(type(img), Image.Image)):
            img = Image.fromarray(img)
        if not(issubclass(type(gt), Image.Image)):
            gt = Image.fromarray(gt)

        if np.random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:, 3]
            xmax = w - bbx[:, 1]
            bbx[:, 1] = xmin
            bbx[:, 3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, gt
        return img, gt, bbx


class Scale(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return torch.from_numpy(np.array(img)) * self.factor
