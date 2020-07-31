from math import cos, sin, pi
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import cuts_segment, draw_centered_text, draw_polygon

TO_COORDINATES_CONVERSION_MATRIX = np.array(
    [
        [cos(-pi / 6) - cos(7 * pi / 6), cos(pi / 2) - cos(7 * pi / 6), cos(7 * pi / 6)],
        [sin(-pi / 6) - sin(7 * pi / 6), sin(pi / 2) - sin(7 * pi / 6), sin(7 * pi / 6)],
        [0, 0, 1],
    ]
)
TO_RGB_CONVERSION_MATRIX = np.linalg.inv(TO_COORDINATES_CONVERSION_MATRIX)


class RGBPoint:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_coordinates(cls, x: float, y: float,
                         intensity: float,
                         accept_out_of_bounds: bool = False) -> Optional['RGBPoint']:
        color = np.dot(TO_RGB_CONVERSION_MATRIX, np.array([[x], [y], [1]]))
        r = color[0, 0]
        g = color[1, 0]
        b = 1 - r - g

        if accept_out_of_bounds:
            r = min(1., max(0., r))
            g = min(1., max(0., g))
            b = min(1., max(0., b))

        if accept_out_of_bounds or (0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1):
            return RGBPoint(r=int(r * intensity * 255),
                            g=int(g * intensity * 255),
                            b=int(b * intensity * 255))
        else:
            return None

    @classmethod
    def from_json(cls, json_data: Dict[str, int]) -> 'RGBPoint':
        return RGBPoint(r=json_data['r'], g=json_data['g'], b=json_data['b'])

    def to_json(self) -> Dict[str, int]:
        return {'r': self.r, 'g': self.g, 'b': self.b}

    @property
    def intensity(self) -> float:
        return (self.r + self.g + self.b) / (256 * 3)

    @property
    def norm_r(self) -> float:
        return self.r / (self.r + self.g + self.b)

    @property
    def norm_g(self) -> float:
        return self.g / (self.r + self.g + self.b)

    @property
    def norm_b(self) -> float:
        return self.b / (self.r + self.g + self.b)

    @property
    def coordinates(self) -> Tuple[float, float]:
        color = np.dot(TO_COORDINATES_CONVERSION_MATRIX, np.array([[self.norm_r], [self.norm_g], [1]]))
        x = color[0, 0]
        y = color[1, 0]
        return float(x), float(y)


EMPTY_PALETTE = [
    [
        RGBPoint.from_coordinates(x=x / 255, y=y / 255, intensity=1)
        for x in range(-255, 256)
    ]
    for y in range(255, -256, -1)
]
EMPTY_PALETTE = [
    [
        [point.norm_b, point.norm_g, point.norm_r] if point is not None
        else [0, 0, 0] if (i-255)**2 + (j-255)**2 <= 255**2
        else [1, 1, 1]
        for i, point in enumerate(row)
    ]
    for j, row in enumerate(EMPTY_PALETTE)
]
EMPTY_PALETTE = np.array(EMPTY_PALETTE)


class RGBRegion:

    def __init__(self, points: List[RGBPoint], size: int = 512, intensity_bar_height: int = 40,
                 intensity_bar_padding: int = 10):
        self._points = [point for point in points if point is not None]

        if len(self._points) == 0:
            self._min_intensity = 0.
            self._max_intensity = 1.
        else:
            intensities = [point.intensity for point in self._points]
            self._min_intensity = min(intensities)
            self._max_intensity = max(intensities)

        self._size = size
        self._intensity_bar_height = intensity_bar_height
        self._intensity_bar_padding = min(intensity_bar_padding, int(intensity_bar_height / 2) - 1)

    @classmethod
    def from_json(cls, json_data: List[Dict[str, int]]) -> 'RGBRegion':
        return RGBRegion(points=[RGBPoint.from_json(point_data) for point_data in json_data])

    def add_point(self, point: Optional[RGBPoint]):
        if point is None:
            return
        self._min_intensity = min(self._min_intensity, point.intensity)
        self._max_intensity = max(self._max_intensity, point.intensity)
        self._points.append(point)
        if len(self._points) == 1:
            self._points.append(RGBPoint(r=point.r, g=point.g, b=point.b))

    def to_json(self) -> List[Dict[str, int]]:
        return [point.to_json() for point in self.points]

    def contains_color(self, r: int, g: int, b: int):
        if len(self.points) <= 2:
            return False

        # Intensity range must be correct.
        point = RGBPoint(r=r, g=g, b=b)
        if not (self._min_intensity <= point.intensity <= self._max_intensity):
            return False

        start_points = self.points
        end_points = self.points[1:] + self.points[:1]
        cut_count = 0
        x, y = point.coordinates
        for start_point, end_point in zip(start_points, end_points):
            if cuts_segment(x=x, y=y,
                            extreme_1=start_point.coordinates,
                            extreme_2=end_point.coordinates):
                cut_count += 1

        # Odd number of cuts means point belongs to region.
        return cut_count % 2 == 1

    def get_color_palette(self):
        # Get z_axis bar.
        intensity_bar = self._get_intensity_bar(width=self.size, padding=self._intensity_bar_padding,
                                                height=self._intensity_bar_height)

        # Get main palette.
        main_palette = self.get_rgb_palette(size=self.size)

        # Concatenate images to obtain output image.
        return np.concatenate([main_palette, intensity_bar], axis=0)

    def replace_last_point(self, point: Optional[RGBPoint]):
        if point is None:
            return
        if len(self._points) >= 1:
            self._points[-1] = point

    def get_rgb_palette(self, size: int) -> np.ndarray:
        palette = EMPTY_PALETTE.copy()
        palette = palette * self.mean_intensity * 255
        palette = palette.astype(np.uint8)
        palette = cv2.resize(palette, dsize=(size, size))

        # draw region and return.
        if len(self._points) >= 2:
            coordinates_list = [point.coordinates for point in self._points]
            points = [((1 + x) / 2, (1 - y) / 2) for x, y in coordinates_list]
            palette = draw_polygon(palette,
                                   points=points,
                                   color=(255, 255, 255),
                                   thickness=2)
        return palette

    def _get_intensity_bar(self,
                           height: int = 20, width: int = 512,
                           padding: int = 10,
                           cursor_width: int = 2) -> np.ndarray:
        # Main bar.
        bar = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
        # Cursor bar.
        bar = cv2.rectangle(img=bar,
                            pt1=(padding, int(height / 2)),
                            pt2=(width - padding, int(height / 2)),
                            color=(0, 0, 0),
                            thickness=-1)

        # Draw vertical cursors.
        bar = self._draw_cursor(bar=bar, padding=padding, cursor_width=cursor_width, intensity=self._min_intensity)
        bar = self._draw_cursor(bar=bar, padding=padding, cursor_width=cursor_width, intensity=self._max_intensity)

        # Write intensities.
        bar = draw_centered_text(bar,
                                 text=f'{self.min_intensity:.2f}',
                                 cx=(width - 2 * padding) * self.min_intensity / width + padding / width)
        bar = draw_centered_text(bar,
                                 text=f'{self.max_intensity:.2f}',
                                 cx=(width - 2 * padding) * self.max_intensity / width + padding / width)
        bar = draw_centered_text(bar,
                                 text=f'{self.mean_intensity:.2f}',
                                 cx=(width - 2 * padding) * self.mean_intensity / width + padding / width)
        return bar

    def reset(self):
        self._min_intensity = 0
        self._max_intensity = 1.
        self._points = []

    @property
    def points(self) -> List[RGBPoint]:
        return self._points[:]

    @property
    def n_points(self) -> int:
        return len(self._points)

    @property
    def size(self) -> int:
        return self._size

    @property
    def mean_intensity(self) -> float:
        return (self._min_intensity + self._max_intensity) / 2

    @property
    def min_intensity(self) -> float:
        return self._min_intensity

    @min_intensity.setter
    def min_intensity(self, intensity: float):
        self._min_intensity = min(1., max(0., min(intensity, self._max_intensity)))

    @property
    def max_intensity(self) -> float:
        return self._max_intensity

    @max_intensity.setter
    def max_intensity(self, intensity: float):
        self._max_intensity = min(1., max(0., max(intensity, self._min_intensity)))

    @property
    def padding(self) -> int:
        return self._intensity_bar_padding

    @staticmethod
    def _draw_cursor(bar: np.ndarray, padding: int, cursor_width: int, intensity: float) -> np.ndarray:
        h, w, _ = bar.shape
        color = int(intensity * 255)
        bar = cv2.rectangle(
            img=bar,
            pt1=(padding + int((w - 2 * padding) * intensity - cursor_width / 2), 0),
            pt2=(padding + int((w - 2 * padding) * intensity + cursor_width / 2), h),
            color=(color, color, color),
            thickness=-1
        )
        return cv2.rectangle(
            img=bar,
            pt1=(padding + int((w - 2 * padding) * intensity - cursor_width / 2), 0),
            pt2=(padding + int((w - 2 * padding) * intensity + cursor_width / 2), h),
            color=(255 - color, 255 - color, 255 - color),
            thickness=1
        )
