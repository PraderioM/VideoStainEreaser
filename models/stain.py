from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull

from models.frame_region import FramePoint, FrameRegion
from models.rgb_region import RGBPoint, RGBRegion
from models.pixel import Pixel


class Stain:
    def __init__(self, frame_region: FrameRegion, rgb_region: RGBRegion):
        self.frame_region = frame_region
        self.rgb_region = rgb_region

    @classmethod
    def from_json(cls, json_data: Dict[str, Union[List[Dict[str, int]], List[Dict[str, float]]]]) -> 'Stain':
        return Stain(frame_region=FrameRegion.from_json(json_data['frame_region']),
                     rgb_region=RGBRegion.from_json(json_data['rgb_region']))

    @classmethod
    def from_pixels(cls, pixels: List[Tuple[FramePoint, RGBPoint]]) -> 'Stain':
        intensities = [point.intensity for _, point in pixels]
        min_intensity = min(intensities)
        max_intensity = max(intensities)
        convex_hull = ConvexHull(np.array([list(point.coordinates) for _, point in pixels]))
        rgb_region = RGBRegion(
            points=[
                RGBPoint.from_coordinates(x=vertex[0],
                                          y=vertex[1],
                                          intensity=min_intensity if i % 2 == 0 else max_intensity)
                for i, vertex in enumerate(convex_hull.vertices)
            ]
        )
        convex_hull = ConvexHull(np.array([[point.x, point.y] for point, _ in pixels]))
        frame_region = FrameRegion(points=[FramePoint(x=vertex[0], y=vertex[1]) for vertex in convex_hull.vertices])

        return Stain(frame_region=frame_region, rgb_region=rgb_region)

    def to_json(self) -> Dict[str, Union[List[Dict[str, int]], List[Dict[str, float]]]]:
        return {
            'frame_region': self.frame_region.to_json(),
            'rgb_region': self.rgb_region.to_json(),
        }

    def draw_on_image(self, image: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        return self.frame_region.draw_on_image(image, color=color, thickness=thickness)

    def get_color_palette(self) -> np.ndarray:
        return self.rgb_region.get_color_palette()

    def get_region_pixels(self, h: int, w: int) -> List[Pixel]:
        return self.frame_region.get_pixels(h=h, w=w)

    def get_stain_pixels(self, image: np.ndarray, region_pixels: Optional[List[Pixel]] = None) -> List[Pixel]:
        # Get missing data from image.
        if region_pixels is None:
            h, w, _ = image.shape
            region_pixels = self.get_region_pixels(h=h, w=w)

        # Stain pixels are formed by those pixels in the frame region that belong to the rgb region.
        stain_pixels: List[Pixel] = []
        for y, x in region_pixels:
            b, g, r = image[y, x, :]
            if self.rgb_region.contains_color(r=r, g=g, b=b):
                stain_pixels.append((y, x))

        return stain_pixels

    def get_averaging_pixels(self, image: np.ndarray,
                             region_pixels: Optional[List[Pixel]] = None,
                             stain_pixels: Optional[List[Pixel]] = None) -> List[Pixel]:
        # Get missing data from image.
        if region_pixels is None:
            h, w, _ = image.shape
            region_pixels = self.get_region_pixels(h=h, w=w)

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
            region_pixels = self.get_region_pixels(h=h, w=w)

            if stain_pixels is None:
                stain_pixels = self.get_stain_pixels(image=image, region_pixels=region_pixels)

            if averaging_pixels is None:
                averaging_pixels = self.get_averaging_pixels(image=image, region_pixels=region_pixels,
                                                             stain_pixels=stain_pixels)

        # Get average color by averaging colors in averaging pixels.
        color_list = np.array([image[y, x, :].tolist() for y, x in averaging_pixels], dtype=np.int)
        average_color = np.mean(color_list, axis=0).astype(image.dtype)

        # Replace averaged color for every stain pixel.
        for y, x in stain_pixels:
            image[y, x, :] = average_color
        return image
