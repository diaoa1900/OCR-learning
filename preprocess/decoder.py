import json

import cv2
import numpy as np


class ImageDecoder(object):
    def __init__(self, img_mode='RGB', channel_first=False):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        # 1、图像解码
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)

        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # RGB在矩形(图像)中的顺序是B,G,R
        if self.img_mode == 'RGB':
            assert img.shape[2] == 3
            img = img[:, :, ::-1]
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        # 2、解码后的图像放入字典
        data['img'] = img
        return data


class LabelDecoder(object):
    def __init__(self):
        pass

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        box_number = len(label)
        points, texts, ignore = [], [], []
        for i in range(0, box_number):
            points.append(label[i]['points'])
            texts.append(label[i]['transcription'])
            if label[i]['transcription'] in ['*', '###']:
                ignore.append(True)
            else:
                ignore.append(False)
        if len(points) == 0:
            return None
        points = self.expand_points_num(points)
        points = np.array(points, dtype=np.float32)
        ignore = np.array(ignore, dtype=np.bool)

        data['polys'] = points
        data['text'] = texts
        data['ignore'] = ignore
        return data

    # 作用是什么啊不懂
    def expand_points_num(self, points):
        max_points_num = 0
        for box in points:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in points:
            ex_box = box + [[box[-1]]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes
