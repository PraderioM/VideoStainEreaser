from itertools import product
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.pixel import Pixel
from utils import draw_polygon, cuts_segment


class FramePoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @classmethod
    def from_json(cls, json_data: Dict[str, float]) -> 'FramePoint':
        return FramePoint(x=json_data['x'], y=json_data['y'])

    @classmethod
    def from_roi(cls, x: float, y: float,
                 x_min: float, y_min: float,
                 x_max: float, y_max: float) -> 'FramePoint':
        w = x_max - x_min
        h = y_max - y_min
        return FramePoint(x=x * w + x_min, y=y * h + y_min)

    def to_json(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y}

    def to_roi(self,
               x_min: float, y_min: float,
               x_max: float, y_max: float) -> 'FramePoint':
        w = x_max - x_min
        h = y_max - y_min
        return FramePoint(x=(self.x - x_min) / w, y=(self.y - y_min) / h)


class FrameRegion:
    def __init__(self, points: List[FramePoint]):
        self._points = points[:]

    @classmethod
    def from_json(cls, json_data: List[Dict[str, float]]) -> 'FrameRegion':
        return FrameRegion(points=[FramePoint.from_json(point_data) for point_data in json_data])

    def to_json(self) -> List[Dict[str, float]]:
        return [point.to_json() for point in self.points]

    def add_point(self, point: FramePoint):
        self._points.append(point)

    def get_pixels(self, h: int, w: int, possible_pixels: Optional[List[Pixel]] = None) -> List[Pixel]:
        if self.n_points <= 2:
            return []

        if possible_pixels is None:
            possible_pixels = [(y, x) for y, x in product(list(range(h)), list(range(w)))]
        pixels: List[Pixel] = []
        for y, x in possible_pixels:
            if self.contains_point(x=x / w, y=y / h):
                pixels.append((y, x))

        return pixels

    def contains_point(self, x: float, y: float) -> bool:
        if self.n_points <= 2:
            return False

        start_points = self.points
        end_points = self.points[1:] + self.points[:1]
        cut_count = 0
        for start_point, end_point in zip(start_points, end_points):
            if cuts_segment(x=x, y=y,
                            extreme_1=(start_point.x, start_point.y),
                            extreme_2=(end_point.x, end_point.y)):
                cut_count += 1

        # Odd number of cuts means point belongs to region.
        return cut_count % 2 == 1

    def draw_on_image(self, image: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        if self.n_points < 2:
            return image
        return draw_polygon(image=image,
                            points=[(point.x, point.y) for point in self.points],
                            color=color, thickness=thickness)

    def draw_on_roi(self, image: np.ndarray,
                    x_min: float, y_min: float,
                    x_max: float, y_max: float,
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
        if self.n_points < 2:
            return image

        return draw_polygon(image=image,
                            points=[
                                (
                                    0 if x_max == x_min else (point.x - x_min) / (x_max - x_min),
                                    0 if y_max == y_min else (point.y - y_min) / (y_max - y_min),
                                )
                                for point in self.points
                            ],
                            color=color, thickness=thickness)

    def get_roi_crop(self, image: np.ndarray, out_h: Optional[int] = None, out_w: Optional[int] = None) -> np.ndarray:
        if self.n_points <= 1:
            if out_h is not None and out_w is not None:
                return np.zeros(shape=(out_h, out_w, 3), dtype=image.dtype)
            else:
                return image

        h, w, _ = image.shape
        x_min = min(w - 2, max(0, int(self.x_min * w)))
        x_max = max(x_min + 1, int(self.x_max * w))
        y_min = min(h - 2, max(0, int(self.y_min * h)))
        y_max = max(y_min + 1, int(self.y_max * h))
        crop = image[y_min: y_max, x_min: x_max].copy()

        if out_h is not None or out_w is not None:
            h, w, _ = crop.shape
            out_h = h if out_h is None else out_h
            out_w = h if out_w is None else out_w
            crop = cv2.resize(crop, dsize=(out_w, out_h))

        return crop

    def replace_last_point(self, point: FramePoint):
        if self.n_points >= 2:
            self._points[-1] = point

    @property
    def points(self) -> List[FramePoint]:
        return self._points[:]

    @property
    def x_min(self) -> float:
        if self.n_points == 0:
            return 0.
        else:
            return min([point.x for point in self.points])

    @property
    def x_max(self) -> float:
        if self.n_points == 0:
            return 1.
        else:
            return max([point.x for point in self.points])

    @property
    def y_min(self) -> float:
        if self.n_points == 0:
            return 0.
        else:
            return min([point.y for point in self.points])

    @property
    def y_max(self) -> float:
        if self.n_points == 0:
            return 1.
        else:
            return max([point.y for point in self.points])

    @property
    def n_points(self) -> int:
        return len(self._points)

    def reset(self):
        self._points = []
