# Below functions are adapted from the Intel OpenVINO toolkit:
# https://github.com/openvinotoolkit/training_extensions (see LICENCE_OPENVINO)
import collections
import numbers
import random

import PIL
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

IMAGE_SIZE = 224
NORM_MEAN_IMGNET = [0.485, 0.456, 0.406]
NORM_STD_IMGNET = [0.229, 0.224, 0.225]

scales = [1.0]
for i in range(1, 4):
    scales.append(scales[-1] * (2 ** (-1 / 4)))


class Compose(object):
    def __init__(self, *args):
        self.children = list(args)

    def __call__(self, sample):
        for child in self.children:
            sample = child(sample)
        return sample

    def randomize_parameters(self):
        for child in self.children:
            if hasattr(child, 'randomize_parameters'):
                child.randomize_parameters()


class ToFloatTensor(object):
    def __call__(self, sample):
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        return torch.from_numpy(sample.copy()).float()


class PermuteImage(object):
    def __call__(self, image):
        return image.permute(2, 0, 1)


class Scale(object):
    """Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and
                                         len(size) == 2)
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Image to be scaled.
        Returns:
            (PIL.Image or np.ndarray): Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = size(img)
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return resize(img, (ow, oh))
        else:
            return resize(img, self.size)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor /= 255.0
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class ColorJitter(object):
    # Based on torchvision color jitter but no hue changes here because it doesn't make a great deal of sense for SLR
    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        img = torchvision.transforms.functional.adjust_brightness(img, self.brightness_factor)
        img = torchvision.transforms.functional.adjust_contrast(img, self.contrast_factor)
        img = torchvision.transforms.functional.adjust_saturation(img, self.saturation_factor)
        return np.asarray(img)

    def randomize_parameters(self):
        brightness = [max(0, 1 - self.brightness), 1 + self.brightness]
        self.brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()

        contrast = [max(0, 1 - self.contrast), 1 + self.contrast]
        self.contrast_factor = random.uniform(contrast[0], contrast[1])

        saturation = [max(0, 1 - self.saturation), 1 + self.saturation]
        self.saturation_factor = random.uniform(saturation[0], saturation[1])


def hflip(img):
    if isinstance(img, np.ndarray):
        return np.fliplr(img).copy()
    return img.transpose(Image.FLIP_LEFT_RIGHT)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self._rand < 0.5:
            return hflip(img)
        return img

    def randomize_parameters(self):
        self._rand = random.random()


class CenterCrop(object):
    """Crops the given image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = size(img)
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return crop(img, (x1, y1, x1 + tw, y1 + th))


def resize(img, size):
    if not isinstance(size, (list, tuple)):
        size = (size, size)
    if isinstance(img, np.ndarray):
        return cv2.resize(img, size)
    return img.resize(size, Image.LINEAR)


def crop(img, position):
    x1, y1, x2, y2 = position
    if isinstance(img, np.ndarray):
        return img[y1:y2, x1:x2]
    return img.crop(position)


def size(img):
    if isinstance(img, np.ndarray) or torch.is_tensor(img):
        h, w, c = img.shape
        return w, h
    w, h = img.size
    return w, h


class MultiScaleCrop(object):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao
    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1,
                 interpolation=Image.LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

        self._crop_scale = None
        self._crop_offset = None
        self._num_scales = len(scale_ratios)
        self._num_offsets = 5 if not more_fix_crop else 13

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))  # upper left
        offsets.append((0, 4 * w_off))  # upper right
        offsets.append((4 * h_off, 0))  # lower left
        offsets.append((4 * h_off, 4 * w_off))  # lower right
        offsets.append((2 * h_off, 2 * w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2 * w_off))  # top center
            offsets.append((4 * h_off, 2 * w_off))  # bottom center
            offsets.append((2 * h_off, 0))  # left center
            offsets.append((2 * h_off, 4 * w_off))  # right center

            offsets.append((1 * h_off, 1 * w_off))  # upper left quarter
            offsets.append((1 * h_off, 3 * w_off))  # upper right quarter
            offsets.append((3 * h_off, 1 * w_off))  # lower left quarter
            offsets.append((3 * h_off, 3 * w_off))  # lower right quarter

        return offsets

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(h=self.height, w=self.width, scales=self.scale_ratios,
                                                      fix_crop=self.fix_crop, max_distort=self.max_distort)

    def fillCropSize(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h in range(len(scale_rates)):
            crop_h = int(base_size * scale_rates[h])
            for w in range(len(scale_rates)):
                crop_w = int(base_size * scale_rates[w])
                # append this cropping size into the list
                if np.absolute(h - w) <= self.max_distort:
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def __call__(self, image):
        w, h = size(image)

        crop_size_pairs = self.fillCropSize(h, w)
        crop_height = crop_size_pairs[self._crop_scale][0]
        crop_width = crop_size_pairs[self._crop_scale][1]

        if self.fix_crop:
            offsets = self.fillFixOffset(h, w)
            h_off = offsets[self._crop_offset][0]
            w_off = offsets[self._crop_offset][1]

        x1, y1, x2, y2 = w_off, h_off, w_off + crop_width, h_off + crop_height

        image = crop(image, (x1, y1, x2, y2))
        return resize(image, (self.width, self.height))

    def randomize_parameters(self):
        self._crop_scale = np.random.choice(self._num_scales)
        self._crop_offset = np.random.choice(self._num_offsets)


def _repr_params(**kwargs):
    params = ['{}={}'.format(k, str(v)) for k, v in kwargs.items()]
    return '({})'.format(', '.join(params))


# --- OPENPOSE --- #

class DeleteFlowKeypoints(object):
    def __init__(self, indices):
        """
        Delete keypoints at given indices.
        :param indices: To delete.
        """
        self.indices = indices

    def __call__(self, sample):
        return np.delete(sample, self.indices, axis=0)
