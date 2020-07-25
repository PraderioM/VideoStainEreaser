from typing import Dict, List, Tuple

import numpy as np

from models.pixel import Pixel
from utils import draw_polygon


class FramePoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @classmethod
    def from_json(cls, json_data: Dict[str, float]) -> 'FramePoint':
        return FramePoint(x=json_data['x'], y=json_data['y'])

    def to_json(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y}


class FrameRegion:
    def __init__(self, points: List[FramePoint]):
        assert len(points) >= 2, 'There are not enough points for defining region.'
        self._points = points[:]

    @classmethod
    def from_json(cls, json_data: List[Dict[str, float]]) -> 'FrameRegion':
        return FrameRegion(points=[FramePoint.from_json(point_data) for point_data in json_data])

    def to_json(self) -> List[Dict[str, float]]:
        return [point.to_json() for point in self.points]

    def add_point(self, point: FramePoint):
        self._points.append(point)

    def remove_last_point(self, point: FramePoint):
        self._points.append(point)

    def get_pixels(self, h: int, w: int) -> List[Pixel]:
        pixels: List[Pixel] = []
        for y in range(h):
            for x in range(w):
                if self.contains_point(x=x / w, y=y / h):
                    pixels.append((y, x))

        return pixels

    def contains_point(self, x: float, y: float) -> bool:
        if len(self.points) == 2:
            return False

        start_points = self.points
        end_points = self.points[1:] + self.points[:1]
        cut_count = 0
        for start_point, end_point in zip(start_points, end_points):
            if self._cuts_segment(x=x, y=y, extreme_1=start_point, extreme_2=end_point):
                cut_count += 1

        # Odd number of cuts means point belongs to region.
        return cut_count % 2 == 1

    @staticmethod
    def _cuts_segment(x: float, y: float, extreme_1: FramePoint, extreme_2: FramePoint):
        x_1, y_1 = extreme_1.x, extreme_1.y
        x_2, y_2 = extreme_2.x, extreme_2.y

        # If segment is vertical we check if point lies exactly in the segment.
        if x_1 == x_2:
            return x_1 == x and y_1 <= y < y

        # If segment is not vertical we get the point vertical projection onto the segment.
        y_aux = (x - x_1) * (y_2 - y_1) / (x_2 - x_1) + y_1

        # The point cuts if the projection falls within the segments limits and above the point.
        return x_1 <= x < x_2 and y_aux >= y

    def draw_on_image(self, image: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        return draw_polygon(image=image, points=self.points, color=color, thickness=thickness)

    @property
    def points(self) -> List[FramePoint]:
        return self._points[:]

    def replace_last_point(self, point: FramePoint):
        self._points[-1] = point
