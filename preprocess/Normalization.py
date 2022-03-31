import numpy as np


class NormalizeImage(object):

    def __init__(self, scale=None, mean=None, std=None, order='channel_height_weight'):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        shape = (3, 1, 1) if order == 'channel_height_weight' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        data['image'] = (data['image'].astype('float32') * self.scale - self.mean) / self.std
        return data
