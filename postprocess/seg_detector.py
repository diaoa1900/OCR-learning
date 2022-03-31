import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class SegDetector(object):
    def __init__(self, traditional_binary_threshold=0.3, score_threshold=0.7, max_number_box=1000, scale_ratio=1.5):
        self.min_pixel = 3
        self.traditional_binary_threshold = traditional_binary_threshold
        self.score_threshold = score_threshold
        self.max_number_box = max_number_box
        self.scale_ratio = scale_ratio

    def __call__(self, batch, predict, out_is_polygon=False):
        p_or_t = predict[:, 0, :, :]
        segmentations = predict > self.traditional_binary_threshold
        boxes_batch = []
        scores_batch = []
        for batch_index in range(p_or_t.shape[0]):
            h, w = batch['shape'][batch_index]
            if out_is_polygon:
                boxes, scores = self.polygon_from_map(p_or_t[batch_index], segmentations[batch_index], h, w)
            else:
                boxes, scores = self.rect_from_map(p_or_t[batch_index], segmentations[batch_index], h, w)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def polygon_from_map(self, p_or_t, segmentation, final_h, final_w):
        predict_h, predict_w = segmentation.shape  # 一张图所以shape即可
        boxes = []
        scores = []
        contours, _ = cv2.findContours((segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_number_box]:
            # 将相对光滑的曲线转为直线
            epsilon = 0.005 * cv2.arcLength(contour, True)  # 0.005 * 周长
            approx = cv2.approxPolyDP(contour, epsilon, True)  # True表示封闭图形
            points = approx.reshape([-1, 2])
            # 框住文字的形状至少是四边形
            if points.shape[0] < 4:
                continue
            score = self.acculate_score(p_or_t, contour.squeeze(1))
            if score < self.score_threshold:
                continue
            box = self.expand_contour(points, self.scale_ratio)
            box = box.reshape([-1, 2])
            _, side_length = self.get_min_box(box.reshape([-1, 1, 2]))
            # 文字框太小了就不要了
            if side_length < self.min_pixel + 2:
                continue
            # 坐标复原到原图大小上
            box[:, 0] = np.clip(np.round(box[:, 0] / predict_h * final_h), 0, final_h)
            box[:, 1] = np.clip(np.round(box[:, 1] / predict_w * final_w), 0, final_w)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def acculate_score(self, p_or_t, contour):
        # 计算文字框内有文本的概率
        h, w = p_or_t.shape
        box = contour.copy()
        # np.clip(a, min, max) 若a小于min,将a置为min，若a大于max，将a置为max，否则不变
        # np.clip  向下取整    np.ceil 向上取整
        x_min = np.clip(np.floor(box[:, 0]).min().astype(np.int), 0, w - 1)
        y_min = np.clip(np.floor(box[:, 1]).min().astype(np.int), 0, h - 1)
        x_max = np.clip(np.ceil(box[:, 0]).max().astype(np.int), 0, w - 1)
        y_max = np.clip(np.ceil(box[:, 1]).max().astype(np.int), 0, h - 1)
        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - x_min
        box[:, 1] = box[:, 1] - y_min
        # cv2.fillPoly 根据提供的点连接成多边形并将其填充为白色
        cv2.fillPoly(mask, box.reshape((1, -1, 2)).astype(np.int32), 1)
        return cv2.mean(p_or_t[y_min:y_max, x_min:x_max], mask)[0]

    def expand_contour(self, box, ratio):
        poly = Polygon(box)
        distance = poly.area * ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expand = np.array(offset.Execute(distance))
        return expand

    def get_min_box(self, box):
        bounding_box = cv2.minAreaRect(box)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[0][1] > points[1][1]:
            index_1 = 1
            index_2 = 0
        if points[2][1] > points[3][1]:
            index_3 = 3
            index_4 = 2
        # 顺时针排列
        box = [points[index_1], points[index_3], points[index_4], points[index_2]]
        return box, min(bounding_box[1])  # bounding_box[1]是矩形的长和宽

