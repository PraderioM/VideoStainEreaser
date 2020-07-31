from typing import Dict, List, Tuple

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
            if cuts_segment(x=x, y=y,
                            extreme_1=(start_point.x, start_point.y),
                            extreme_2=(end_point.x, end_point.y)):
                cut_count += 1

        # Odd number of cuts means point belongs to region.
        return cut_count % 2 == 1

    def draw_on_image(self, image: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        return draw_polygon(image=image,
                            points=[(point.x, point.y) for point in self.points],
                            color=color, thickness=thickness)

    @property
    def points(self) -> List[FramePoint]:
        return self._points[:]

    def replace_last_point(self, point: FramePoint):
        self._points[-1] = point
