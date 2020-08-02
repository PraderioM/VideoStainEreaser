from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from models.frame_region import FramePoint, FrameRegion
from models.pixel import Pixel


class Stain:
    def __init__(self, neighborhood: FrameRegion, crop: FrameRegion):
        self.neighborhood = neighborhood
        self.crop = crop

    @classmethod
    def from_json(cls, json_data: Dict[str, List[Dict[str, float]]]) -> 'Stain':
        return Stain(neighborhood=FrameRegion.from_json(json_data['neighborhood']),
                     crop=FrameRegion.from_json(json_data['crop']))

    @classmethod
    def from_pixels(cls, pixels: List[Tuple[FramePoint, bool]]) -> 'Stain':
        convex_hull = ConvexHull(np.array([[point.x, point.y] for point, _ in pixels]))
        frame_region_neighborhood = FrameRegion(points=[FramePoint(x=vertex[0], y=vertex[1])
                                                        for vertex in convex_hull.vertices])
        convex_hull = ConvexHull(np.array([[point.x, point.y] for point, is_precise in pixels if is_precise]))
        precise_frame_region = FrameRegion(points=[FramePoint(x=vertex[0], y=vertex[1])
                                                   for vertex in convex_hull.vertices])

        return Stain(neighborhood=frame_region_neighborhood, crop=precise_frame_region)

    def to_json(self) -> Dict[str, List[Dict[str, float]]]:
        return {
            'neighborhood': self.neighborhood.to_json(),
            'crop': self.crop.to_json(),
        }

    def draw_on_image(self, image: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        return self.neighborhood.draw_on_image(image, color=color, thickness=thickness)

    def get_roi_crop(self, image: np.ndarray, out_h: Optional[int] = None, out_w: Optional[int] = None) -> np.ndarray:
        return self.neighborhood.get_roi_crop(image, out_h=out_h, out_w=out_w)

    def draw_on_roi(self, image: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float,
                    color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        return self.crop.draw_on_roi(image, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                                     color=color, thickness=thickness)

    def get_neighborhood_pixels(self, h: int, w: int) -> List[Pixel]:
        return self.neighborhood.get_pixels(h=h, w=w)

    def get_stain_pixels(self, image: np.ndarray, region_pixels: Optional[List[Pixel]] = None) -> List[Pixel]:
        # Get missing data from image.
        h, w, _ = image.shape
        if region_pixels is None:
            region_pixels = self.get_neighborhood_pixels(h=h, w=w)

        return self.crop.get_pixels(h=h, w=w, possible_pixels=region_pixels)

    def get_averaging_pixels(self, image: np.ndarray,
                             region_pixels: Optional[List[Pixel]] = None,
                             stain_pixels: Optional[List[Pixel]] = None) -> List[Pixel]:
        # Get missing data from image.
        if region_pixels is None:
            h, w, _ = image.shape
            region_pixels = self.get_neighborhood_pixels(h=h, w=w)

        if stain_pixels is None:
            stain_pixels = self.get_stain_pixels(image=image, region_pixels=region_pixels)

        # Averaging pixels are pixels in region that are not stain pixels.
        averaging_pixels: List[Pixel] = []
        for y, x in region_pixels:
            is_stain_pixel = False
            for stain_y, stain_x in stain_pixels:
                if stain_y == y and stain_x == x:
                    is_stain_pixel = True
                    break

            if not is_stain_pixel:
                averaging_pixels.append((y, x))
        return averaging_pixels

    def fill(self, image: np.ndarray,
             stain_pixels: Optional[List[Pixel]] = None,
             averaging_pixels: Optional[List[Pixel]] = None) -> np.ndarray:
        # Get missing data from image.
        if stain_pixels is None or averaging_pixels is None:
            h, w, _ = image.shape
            region_pixels = self.get_neighborhood_pixels(h=h, w=w)

            if stain_pixels is None:
                stain_pixels = self.get_stain_pixels(image=image, region_pixels=region_pixels)

            if averaging_pixels is None:
                averaging_pixels = self.get_averaging_pixels(image=image, region_pixels=region_pixels,
                                                             stain_pixels=stain_pixels)

        # If there are no averaging pixels we can average nothing.
        if len(averaging_pixels) == 0:
            return image

        # Get average color by averaging colors in averaging pixels.
        color_list = np.array([image[y, x, :].tolist() for y, x in averaging_pixels], dtype=np.int)
        average_color = np.mean(color_list, axis=0).astype(image.dtype)

        # Replace averaged color for every stain pixel.
        for y, x in stain_pixels:
            image[y, x, :] = average_color
        return image

    @property
    def x_min(self) -> float:
        return self.neighborhood.x_min

    @property
    def x_max(self) -> float:
        return self.neighborhood.x_max

    @property
    def y_min(self) -> float:
        return self.neighborhood.y_min

    @property
    def y_max(self) -> float:
        return self.neighborhood.y_max
