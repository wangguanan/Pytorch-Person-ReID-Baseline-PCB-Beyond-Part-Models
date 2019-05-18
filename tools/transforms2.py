import torch
import torchvision.transforms.functional as F
import random


def fliplr(img, device=None):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if device is not None:
        inv_idx = inv_idx.to(device)
    img_flip = img.index_select(3,inv_idx)
    return img_flip


class MisAlgnAug(object):

    def __init__(self, crop_prob=0.5,  ratio=0.1):
        self.crop_prob = crop_prob
        self.ratio = ratio

    def __call__(self, img):
        '''
        :param img: PIL Image to be cropped
        :return: PIL Image
        '''
        is_crop = random.uniform(0,1) < self.crop_prob
        position = random.choice(['up', 'bottom'])
        operation = random.choice(['crop', 'pad'])
        ratio = self.ratio

        if is_crop:
            w, h = img.size
            th = int(h * ratio)
            if position == 'up' and operation == 'crop':
                return F.crop(img, th, 0, h-th, w)
            elif position == 'bottom' and operation == 'crop':
                return F.crop(img, 0, 0, h-th, w)
            elif position == 'up' and operation == 'pad':
                return F.pad(img, (0, th, 0, 0), padding_mode='edge')
            elif position == 'bottom' and operation == 'pad':
                return F.pad(img, (0, 0, 0, th), padding_mode='edge')
        else:
            return img