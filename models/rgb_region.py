from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from utils import draw_centered_text


class RGBPoint:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_json(cls, json_data: Dict[str, int]) -> 'RGBPoint':
        return RGBPoint(r=json_data['r'], g=json_data['g'], b=json_data['b'])

    def to_json(self) -> Dict[str, int]:
        return {'r': self.r, 'g': self.g, 'b': self.b}


class RGBRegion:
    POSSIBLE_AXES = {'r', 'g', 'b'}

    def __init__(self, points: List[RGBPoint], size: int = 512, x_axis: str = 'r', y_axis: str = 'g',
                 z_axis_val: int = 0, z_bar_height: int = 20,
                 z_sections_size: int = 96):
        rgb_points: List[RGBPoint] = []
        for point in points:
            for r in range(-1, 2):
                for g in range(-1, 2):
                    for b in range(-1, 2):
                        rgb_points.append(RGBPoint(r=point.r+r, g=point.g+g, b=point.b+b))
        if len(rgb_points) == 0:
            self._convex_hull = None
        else:
            self._convex_hull = ConvexHull(np.array([[point.r, point.g, point.b] for point in rgb_points]))
        self._size = size

        assert x_axis in self.POSSIBLE_AXES, f'Unrecognized axis value `{x_axis}`'
        assert y_axis in self.POSSIBLE_AXES, f'Unrecognized axis value `{y_axis}`'
        error_msg = f'x_axis and y_axis values must be different however got x_axis = {x_axis} and y_axis = {y_axis}.'
        assert x_axis != y_axis, error_msg
        self._x_axis = x_axis
        self._y_axis = y_axis
        self.z_axis_val = z_axis_val
        self._z_bar_height = z_bar_height
        self._z_sections_size = z_sections_size

    @classmethod
    def from_json(cls, json_data: List[Dict[str, int]]) -> 'RGBRegion':
        return RGBRegion(points=[RGBPoint.from_json(point_data) for point_data in json_data])

    def add_point(self, point: RGBPoint):
        if self._convex_hull is None:
            self._convex_hull = ConvexHull(np.array([[point.r, point.g, point.b]], dtype=np.int))
        else:
            points = self._convex_hull.vertices
            self._convex_hull = ConvexHull(
                np.concatenate([points, np.array([[point.r, point.g, point.b]],
                                                 dtype=points.dtype)])
            )

    def to_json(self) -> List[Dict[str, int]]:
        return [point.to_json() for point in self.points]

    def is_color_in_region(self, r: int, g: int, b: int, hull: Optional[ConvexHull] = None):
        if hull is None:
            convex_hull = self.convex_hull

            if convex_hull is None:
                return False
            else:
                hull = convex_hull

        points = hull.points
        new_hull = ConvexHull(np.concatenate([points, np.array([[r, g, b]], dtype=points.dtype)]))
        return np.array_equal(new_hull.vertices, hull.vertices)

    @property
    def convex_hull(self) -> Optional[ConvexHull]:
        return self._convex_hull

    def get_color_palette(self):
        # Get z_axis bar.
        z_axis_bar = self._get_z_axis_bar(axis=self.z_axis, val=self.z_axis_val, width=self.size,
                                          height=self.z_bar_height)

        # Get main palette.
        main_palette = self.get_z_section_palette(size=self.size, x_axis=self.x_axis, y_axis=self.y_axis,
                                                  z_axis_val=self.z_axis_val)

        # Get auxiliary palettes.
        auxiliary_palettes = self._get_auxiliary_palettes(size=self.z_sections_size, height=self.size,
                                                          x_axis=self.x_axis, y_axis=self.y_axis)

        # Get clear button.
        clear_button = self._get_clear_button(width=self.z_sections_size, height=self.z_bar_height)

        # Concatenate images to obtain output image.
        return np.concatenate(arrays=[
            np.concatenate([main_palette, z_axis_bar], axis=0),
            np.concatenate([auxiliary_palettes, clear_button], axis=0)
        ],
            axis=1)

    def get_z_section_palette(self, size: int, x_axis: str, y_axis: str, z_axis_val: int,
                              color_shift: int = 5, n_pixels_per_color: int = 4) -> np.ndarray:
        z_axis = list(self.POSSIBLE_AXES - {x_axis, y_axis})[0]
        convex_hull = self.convex_hull
        z_section: List[List[List[int]]] = []
        for y in range(255, -1, -1):
            band: List[List[List[int]]] = [[] for _ in range(n_pixels_per_color)]
            for x in range(256):
                color = {x_axis: x, y_axis: y, z_axis: z_axis_val}
                if not self.is_color_in_region(r=color['r'], g=color['g'], b=color['b'], hull=convex_hull):
                    for i in range(n_pixels_per_color):
                        for j in range(n_pixels_per_color):
                            if x * n_pixels_per_color + i + y * n_pixels_per_color + j % 2 == 0:
                                shift_val = color_shift
                            else:
                                shift_val = -color_shift
                            new_color = {axis: min(255, max(0, val + shift_val)) for axis, val in color.items()}
                            band[j].append([new_color['b'], new_color['g'], new_color['r']])

            # Add new band.
            z_section.extend(band)

        # Resize and return.
        image = np.array(z_section, dtype=np.uint8)
        return cv2.resize(image, dsize=(size, size))

    def _get_auxiliary_palettes(self, size: int, height: int, x_axis: str, y_axis: str) -> np.ndarray:
        n_palettes = height // size
        remaining_blank_height = height - n_palettes * size

        palette_list: List[np.ndarray] = []
        for i in range(n_palettes):
            # Blank image.
            blank_height = remaining_blank_height // (n_palettes + 1 - i)
            blank_image = np.ones((blank_height, size, 3)) * 255
            remaining_blank_height = remaining_blank_height - blank_height

            # Auxiliary palette.
            z_axis_val = 125 if n_palettes == 1 else int(255 * i / (n_palettes - 1))
            auxiliary_palette = self.get_z_section_palette(size=size, x_axis=x_axis, y_axis=y_axis,
                                                           z_axis_val=z_axis_val)

            palette_list.append(blank_image)
            palette_list.append(auxiliary_palette)

        # Get last blank image
        palette_list.append(np.ones((remaining_blank_height, size, 3)) * 255)

        return np.concatenate(palette_list, axis=0)

    def _get_z_axis_bar(self, axis: str, val: int,
                        height: int = 20, width: int = 512,
                        pointer_width: int = 2) -> np.ndarray:
        bar_height = int(height/2)
        bar = np.ones(shape=(bar_height, width, 3), dtype=np.uint8) * 255
        bar = cv2.rectangle(img=bar,
                            pt1=(0, int(bar_height / 2)),
                            pt2=(width, int(bar_height / 2)),
                            color=(0, 0, 0),
                            thickness=-1)
        pointer_color = tuple(self.get_axis_color(axis=axis, val=val))
        bar = cv2.rectangle(img=bar,
                            pt1=(int(width * val / 255 - pointer_width / 2), 0),
                            pt2=(int(width * val / 255 + pointer_width / 2), bar_height),
                            color=pointer_color,
                            thickness=-1)

        upper_bar_height = height - bar_height
        upper_bar = np.ones((upper_bar_height, width, 3), dtype=np.uint8) * 255
        upper_bar = draw_centered_text(upper_bar, text='change for x axis', cx=1/3)
        upper_bar = draw_centered_text(upper_bar, text='change for y axis', cx=2/3)

        return np.concatenate([upper_bar, bar], axis=0)

    def change_x_axis(self):
        self._x_axis = self.z_axis

    def change_y_axis(self):
        self._y_axis = self.z_axis

    @property
    def points(self) -> List[RGBPoint]:
        if self._convex_hull is None:
            return []
        else:
            return [RGBPoint(r=vertex[0], g=vertex[1], b=vertex[2]) for vertex in self._convex_hull.vertices]

    @property
    def size(self) -> int:
        return self._size

    @property
    def z_bar_height(self) -> int:
        return self._z_bar_height

    @property
    def z_sections_size(self) -> int:
        return self._z_sections_size

    @property
    def x_axis(self) -> str:
        return self._x_axis

    @property
    def y_axis(self) -> str:
        return self._y_axis

    @property
    def z_axis(self) -> str:
        return list(self.POSSIBLE_AXES - {self._x_axis, self._y_axis})[0]

    @staticmethod
    def get_axis_color(axis: str, val: int = 1) -> np.ndarray:
        if axis == 'r':
            color = (0, 0, val)
        elif axis == 'g':
            color = (0, val, 0)
        elif axis == 'b':
            color = (val, 0, 0)
        else:
            raise ValueError(f'Unrecognized axis `{axis}`.')

        return np.array(color, dtype=np.uint8)

    @staticmethod
    def _get_clear_button(width: int, height: int):
        button = np.ones((height, width, 3), dtype=np.uint8) * 255
        return draw_centered_text(button, text='clear')

    def clear_region(self):
        self._convex_hull = None
