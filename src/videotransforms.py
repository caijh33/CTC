import numpy as np
import numbers
import random
import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
import random
from PIL import ImageOps


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
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

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i + h, j:j + w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
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

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class VideoCrop(object):

    def __init__(self, size):
        self.size = size
        self.window_size = 3

    def __call__(self, imgs):
        '''
        first reshape img into 256(shorter length), then clip 3 256 x 256 img in window. if need to resize to 224 x 224 ?
        :param imgs:
        :return:
        '''
        t, h, w, c = imgs.shape  # batch x 256 x 340 x 3
        th, tw = (self.size, self.size)
        video_imgs = list()
        for n in range(self.window_size):
            x1 = int(round((w - tw) / self.window_size * n))
            y1 = int(round((h - th) / self.window_size * n))
            x2 = x1 + tw
            y2 = y1 + th
            # print(x1, y1, x2, y2)
            img = np.resize(imgs[:, y1:y2, x1:x2, :], (t, th, tw, c))  # all img resize to th, tw ?
            video_imgs.append(img)
        return video_imgs

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        corner_imgs = list()
        for n in self.crop_positions:
            # print(n)
            if n == 'c':
                th, tw = (self.size, self.size)
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif n == 'tl':
                x1 = 0
                y1 = 0
                x2 = self.size
                y2 = self.size
            elif n == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = self.size
            elif n == 'bl':
                x1 = 0
                y1 = h - self.size
                x2 = self.size
                y2 = h
            elif n == 'br':
                x1 = w - self.size
                y1 = h - self.size
                x2 = w
                y2 = h
            corner_imgs.append(imgs[:, y1:y2, x1:x2, :])
        return corner_imgs

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
        # TODO: make efficient
        # for t, m, s in zip(tensor, self.mean, self.std):
        #    t.sub_(m).div_(s)
        xmax, xmin = tensor.max(), tensor.min()
        tensor = (tensor - xmin) / (xmax - xmin)
        return tensor

    def randomize_parameters(self):
        pass


def transform_data(data, scale_size=256, crop_size=224, random_crop=False, random_flip=False):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    if random_crop:
        x0 = random.randint(0, width - crop_size)
        y0 = random.randint(0, height - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    else:
        x0 = int((width - crop_size) / 2)
        y0 = int((height - crop_size) / 2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if random_flip and random.randint(0, 1) == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return data


def get_10_crop(data, scale_size=256, crop_size=224):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    top_left = [[0, 0],
                [width - crop_size, 0],
                [int((width - crop_size) / 2), int((height - crop_size) / 2)],
                [0, height - crop_size],
                [width - crop_size, height - crop_size]]
    crop_data = []
    for point in top_left:
        non_flip = []
        flip = []
        x_0 = point[0]
        y_0 = point[1]
        x_1 = x_0 + crop_size
        y_1 = y_0 + crop_size
        for img in data:
            tmp = img.crop((x_0, y_0, x_1, y_1))
            non_flip.append(tmp)
            flip.append(ImageOps.mirror(tmp))
        crop_data.append(non_flip)
        crop_data.append(flip)
    return crop_data


def scale(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width == scale_size and height >= width) or (height == scale_size and width >= height):
        return data
    if width >= height:
        h = scale_size
        w = round((width / height) * scale_size)
    else:
        w = scale_size
        h = round((height / width) * scale_size)
    for i, image in enumerate(data):
        data[i] = image.resize((w, h))
    return data


def resize(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width == scale_size and height >= width) or (height == scale_size and width >= height):
        return data
    for i, image in enumerate(data):
        data[i] = image.resize((scale_size, scale_size))
    return data


def video_frames_resize(data, scale_size):
    t, h, w, c = data.shape

    if h >= scale_size and w >= scale_size:
        return data
    else:
        data2 = data.copy()
        data2.resize((t, scale_size, scale_size, c))
        return data2

