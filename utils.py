import time
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

def draw(image,mask_true,mask_pred,img_name):
    mask_pred = mask_pred.cpu().detach().numpy().astype(np.uint8)
    mask_true = mask_true.cpu().detach().numpy().astype(np.uint8)
    image = np.squeeze(image.permute(1,2,0).cpu().detach().numpy().astype(np.uint8))
    img_src = image.copy()
    image = SegmentationMapsOnImage(mask_pred, shape=image.shape).draw_on_image(image)[0]
    img_src = SegmentationMapsOnImage(mask_true, shape=img_src.shape).draw_on_image(img_src)[0]
    image = np.hstack((img_src,image))
    cv2.imwrite(img_name,image)

def write_and_print_logs(log_dir,log_content):
    print(log_content)
    with open(log_dir,"a+") as text_log:
        text_log.write(log_content+"\n")

class_indices = {
    "sky" : 0,
    "building" : 1,
    "pole" : 2,
    "road" : 3,
    "pavement" : 4,
    "tree" : 5,
    "signsymbol" : 6,
    "fence": 7,
    "car" : 8,
    "pedestrian" : 9,
    "bicyclist" : 10,
    "unlabelled" : 11,
}

cityscapes_label_map = {
    'road'         :  1,
    'sidewalk'     :  2,
    'building'     :  3,
    'wall'         :  4,
    'fence'        :  5,
    'pole'         :  6,
    'traffic light':  7,
    'traffic sign' :  8,
    'vegetation'   :  9,
    'terrain'      : 10,
    'sky'          : 11,
    'person'       : 12,
    'rider'        : 13,
    'car'          : 14,
    'truck'        : 15,
    'bus'          : 16,
    'train'        : 17,
    'motorcycle'   : 18,
    'bicycle'      : 19,
    'none'         : 0,
}

ddd17_label_map = {
    "flat":0,
    'construction+sky':1,
    'object':2,
    'nature':3,
    'human':4,
    'vehicle':5
}

def get_class_indices(datasets_name):
    if datasets_name in ["CamVid"]:
        return class_indices
    elif datasets_name in ["cityscape"]:
        return cityscapes_label_map
    elif datasets_name in ["ddd17"]:
        return ddd17_label_map
