import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class MakeThresholdMap(object):
    def __init__(self, scale_ratio=0.4, min_threshold=0.3, max_threshold=0.7):
        self.scale_ratio = scale_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def __call__(self, data):
        image = data['image']
        poly = data['polys']
        ignore = data['ignore']

        gt = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(poly)):
            if ignore[i]:
                continue
            self.draw(poly, gt, mask)
        gt = gt * (self.max_threshold - self.min_threshold) + self.min_threshold

        data['threshold_map'] = gt
        data['threshold_mask'] = mask
        return data

    def draw(self, poly, gt, mask):
        poly = np.array(poly)
        assert poly.ndim == 2
        assert poly.shape[1] == 2

        polygon = Polygon(poly)
        if polygon.area <= 0:
            return
        distance = polygon.area * (1 - np.power(self.scale_ratio, 2)) / polygon.length
        subject = [tuple(p) for p in poly]
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expand_polygon = np.array(offset.Execute(distance)[0])
        cv2.fillPoly(mask, expand_polygon.astype(np.int32)[np.newaxis, :, :], 1)

        x_min = expand_polygon[:, 0].min()
        y_min = expand_polygon[:, 1].min()
        x_max = expand_polygon[:, 0].max()
        y_max = expand_polygon[:, 1].max()
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        # np.broadcast_to 将数组广播到新形状           np.linspace 按照指定均匀步长生成数字序列
        xs = np.broadcast_to(np.linspace(0, width-1, 1), (height, width))
        ys = np.broadcast_to(np.linspace(0, height-1, 1).reshape(height, 1), (height, width))
        distance_map = np.zeros((poly.shape[0], height, width), dtype=np.float32)  # 有多少个点就生成几张图
        for i in range(poly.shape[0]):
            j = (i + 1) % poly.shape[0]
            # 计算每个点到两点连成的边的距离
            absolute_distance = self.distance(xs, ys, poly[i], poly[j])
            distance_map[i] = np.clip(absolute_distance/distance, 0, 1)
        # 每个点到各边的几个距离中取最小的
        distance_map = distance_map.min(axis=0)
        x_min_valid = min(max(0, x_min), gt.shape[1]-1)
        y_min_valid = min(max(0, y_min), gt.shape[0]-1)
        x_max_valid = min(max(0, x_max), gt.shape[1]-1)  # ???????????????????????????????
        y_max_valid = min(max(0, y_max), gt.shape[0]-1)  # ???????????????????????????????
        gt[y_min_valid:y_max_valid+1, x_min_valid:x_max_valid+1] = np.fmax(1 - distance_map[y_min_valid-y_min:y_max_valid-y_max+height, x_min_valid-x_min:x_max_valid-x_max+width], gt[y_min_valid:y_max_valid+1, x_min_valid:x_max_valid+1])

    def distance(self, xs, ys, point_1, point_2):
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
                    2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
