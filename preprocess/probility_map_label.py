import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class MakeShrinkMap(object):
    def __init__(self, min_text_pixel=8, scale_ratio=0.4):
        self.min_text_pixel = min_text_pixel
        self.scale_ratio = scale_ratio

    def __call__(self, data):
        image = data['image']
        points = data['polys']
        ignores = data['ignore']
        h, w = image.shape[:2]

        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)

        for i in range(len(points)):
            poly = points[i]
            height = max(poly[:, 1]) - min(poly[:, 1])
            weight = max(poly[:, 0]) - min(poly[:, 0])
            if ignores[i] or min(height, weight) < self.min_text_pixel:
                ignores[i] = True
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                poly = self.shrink_func(poly, self.scale_ratio)
                if poly.size == 0:
                    ignores[i] = True
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                else:
                    cv2.fillPoly(gt, poly.astype(np.int32)[np.newaxis, :, :], 1)
        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def shrink_func(self, poly, ratio):
        polygon = Polygon(poly)
        distance = cv2.contourArea(polygon) * (1 - np.power(ratio, 2)) / polygon.length
        subject = [tuple(p) for p in poly]
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrink = offset.Execute(-distance)
        if not shrink:
            shrink = np.array(shrink)
        else:
            shrink = np.array(shrink[0]).reshape(-1, 2)
        return shrink
