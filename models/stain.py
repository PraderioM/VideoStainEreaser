# from math import sqrt
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
             averaging_pixels: Optional[List[Pixel]] = None,
             per_pixel_averaging_list: Optional[List[Tuple[Pixel, List[Pixel]]]] = None,
             fill_type: int = 1) -> np.ndarray:
        # Get missing data from image.
        complete_fill_1 = stain_pixels is None or averaging_pixels is None
        complete_fill_2 = per_pixel_averaging_list is None and fill_type == 2
        if complete_fill_1 or complete_fill_2:
            if stain_pixels is None or averaging_pixels is None:
                h, w, _ = image.shape
                region_pixels = self.get_neighborhood_pixels(h=h, w=w)

                if stain_pixels is None:
                    stain_pixels = self.get_stain_pixels(image=image, region_pixels=region_pixels)

                if averaging_pixels is None:
                    averaging_pixels = self.get_averaging_pixels(image=image, region_pixels=region_pixels,
                                                                 stain_pixels=stain_pixels)
            if complete_fill_2:
                per_pixel_averaging_list = self.get_per_pixel_averaging_list(stain_pixels=stain_pixels,
                                                                             averaging_pixels=averaging_pixels)

        # If there are no averaging pixels we can average nothing.
        if len(averaging_pixels) == 0:
            return image

        # Get average color by averaging colors in averaging pixels.
        if fill_type == 1:
            return self._fast_fill(image=image, stain_pixels=stain_pixels, averaging_pixels=averaging_pixels)
        elif fill_type == 2:
            return self._medium_fill(image=image, per_pixel_averaging_list=per_pixel_averaging_list)
        else:
            raise ValueError(f'Unrecognized fill type `{fill_type}`.')

    # @staticmethod
    # def get_weighted_stain_pixels(stain_pixels: List[Pixel],
    #                               averaging_pixels: List[Pixel]) -> List[Tuple[Pixel, List[Tuple[float, Pixel]]]]:
    #
    #     distance_stain_pixels = [
    #         (
    #             (y, x),
    #             [
    #                 (
    #                     sqrt((y - n_y)**2 + (x - n_x)**2),
    #                     (n_y, n_x)
    #                 )
    #                 for n_y, n_x in averaging_pixels
    #             ]
    #         )
    #         for y, x in stain_pixels
    #     ]
    #     total_distance_list = [sum([d for d, _ in distances]) for _, distances in distance_stain_pixels]
    #     return [
    #         (
    #             pixel,
    #             [
    #                 (
    #                     d / total_d,
    #                     n_pixel
    #                 )
    #                 for d, n_pixel in distances
    #             ]
    #         )
    #         for (pixel, distances), total_d in zip(distance_stain_pixels, total_distance_list)
    #     ]

    @staticmethod
    def get_per_pixel_averaging_list(stain_pixels: List[Pixel],
                                     averaging_pixels: List[Pixel]) -> List[Tuple[Pixel, List[Pixel]]]:
        per_pixel_averaging_list: List[Tuple[Pixel, List[Pixel]]] = []
        for y, x in stain_pixels:
            averaging_list = [(n_y, n_x) for n_y, n_x in averaging_pixels if n_y == y or n_x == x]
            if len(averaging_list) == 0:
                averaging_list = averaging_pixels[:]
            per_pixel_averaging_list.append(((y, x), averaging_list))
        return per_pixel_averaging_list

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

    @staticmethod
    def _fast_fill(image: np.ndarray, stain_pixels: List[Pixel], averaging_pixels: List[Pixel]) -> np.ndarray:
        color_list = np.array([image[y, x, :].tolist() for y, x in averaging_pixels], dtype=np.int)
        average_color = np.mean(color_list, axis=0).astype(image.dtype)

        # Replace averaged color for every stain pixel.
        for y, x in stain_pixels:
            image[y, x, :] = average_color
        return image

    @staticmethod
    def _medium_fill(image: np.ndarray, per_pixel_averaging_list: List[Tuple[Pixel, List[Pixel]]]) -> np.ndarray:
        # Replace averaged color for every stain pixel.
        for (y, x), averaging_pixels in per_pixel_averaging_list:
            color_list = np.array([image[n_y, n_x, :].tolist() for n_y, n_x in averaging_pixels], dtype=np.int)
            average_color = np.mean(color_list, axis=0).astype(image.dtype)
            image[y, x, :] = average_color
        return image

    # @staticmethod
    # def _slow_fill(image: np.ndarray,
    #                weighted_stain_pixels: List[Tuple[Pixel, List[Tuple[float, Pixel]]]]) -> np.ndarray:
    #
    #     # Replace averaged color for every stain pixel.
    #     for (y, x), neighbour_pixels in weighted_stain_pixels:
    #         color_list = [
    #             (weight * image[n_y, n_x, :].astype(np.float)).tolist()
    #             for weight, (n_y, n_x) in neighbour_pixels
    #         ]
    #         average_color = np.mean(np.array(color_list, dtype=np.float), axis=0).astype(image.dtype)
    #         image[y, x, :] = average_color
    #
    #     return image
