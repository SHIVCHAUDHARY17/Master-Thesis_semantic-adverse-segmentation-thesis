
import collections
import torchvision
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image


class ExtRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.size[1]*scale), int(img.size[0]*scale))
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        return self.__class__.__name__ + '(scale_range={0})'.format(self.scale_range)


class ExtScale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        target_size = (int(img.size[1]*self.scale), int(img.size[0]*self.scale))
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        return self.__class__.__name__ + '(scale={0})'.format(self.scale)


class ExtRandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("Degrees must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("Degrees must be a sequence of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        return random.uniform(degrees[0], degrees[1])

    def __call__(self, img, lbl):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(lbl, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)


class ExtRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser

    def __call__(self, img, lbl):
        h, w = img.size
        ph = (h // 32 + 1) * 32 - h if h % 32 != 0 else 0
        pw = (w // 32 + 1) * 32 - w if w % 32 != 0 else 0
        im = F.pad(img, (pw//2, pw - pw//2, ph//2, ph - ph//2))
        lbl = F.pad(lbl, (pw//2, pw - pw//2, ph//2, ph - ph//2))
        return im, lbl


class ExtToTensor(object):
    def __init__(self, normalize=True, target_type='int64'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, pic, lbl):
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(pic, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(np.array(lbl, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ExtNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        return F.normalize(tensor, self.mean, self.std), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtRandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class ExtColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img, lbl):
        return self.transform(img), lbl

    def __repr__(self):
        return self.__class__.__name__ + '()'

__all__ = [
    'ExtCompose', 'ExtResize', 'ExtRandomCrop', 'ExtCenterCrop', 'ExtRandomScale',
    'ExtScale', 'ExtPad', 'ExtToTensor', 'ExtNormalize', 'ExtRandomHorizontalFlip',
    'ExtRandomVerticalFlip', 'ExtColorJitter', 'ExtRandomRotation'
]

