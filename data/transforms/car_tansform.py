# encoding: utf-8
"""
@author:  liaoxingyu ++
@contact: liaoxingyu2@jd.com ++
"""

import glob
import random
import math
import numpy as np
from PIL import Image
import torch


class ImageErasing(object):
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

    def __init__(self, probability=0.33, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
        self.car_imgs = sorted(glob.glob("cars_/*.*"))
        self.len_car_imgs = len(self.car_imgs)

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            # bbox shape
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img_nb = round(random.uniform(self.len_imgs))
                    car_path = self.car_imgs[img_nb]
                    img_car = np.asarray(Image.open(car_path).convert('RGB')) / 255

                    # cropping borders and upper half
                    img_car_h, img_car_w = img_car.shape[:2]
                    img_car = img_car[img_car_h // 2:img_car_h // 10 * 9, img_car_w // 20:img_car_w // 20 * 19, :]
                    img_car_h, img_car_w = img_car.shape[:2]
                    
                    # cropping random bbox from image
                    y1_car = random.randint(0, img_car_h - h)
                    x1_car = random.randint(0, img_car_w - w)
                    img_car = img_car[y1_car:y1_car + h, x1_car:x1_car + w, :]

                    # patching it to the image
                    img_car = np.transpose(img_car, (2, 0, 1))
                    img[:, x1:x1 + h, y1:y1 + w] = img_car[:, :, :]
                    
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = self.mean[:]
                    
                return img

        return img
