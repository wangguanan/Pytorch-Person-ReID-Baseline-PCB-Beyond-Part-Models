import torch
import torchvision.transforms.functional as F
import random
import math

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

def fliplr(img, device=None):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if device is not None:
        inv_idx = inv_idx.to(device)
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img