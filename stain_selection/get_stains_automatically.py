from typing import List, Optional, Tuple
from itertools import product

import cv2
import numpy as np
from tqdm import tqdm

from models.stain import Stain
from models.frame_region import FramePoint
from models.rgb_region import RGBPoint


def get_stains_automatically(path: str, tol: int = 3) -> List[Stain]:
    video = cv2.VideoCapture(path)
    progress = tqdm(desc='Processing video in look for stains.', total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Initialize loop variables.
    _, prev_image = video.read()
    ret, current_image = video.read()
    progress.update(2)
    h, w, _ = prev_image.shape
    possible_stain_pixels = [(i, j) for i, j in product(range(w), range(h))]

    while ret and len(possible_stain_pixels) > 0:
        # Stain pixels remain unchanged during all video.
        difference = np.abs(current_image.astype(dtype=np.int) - prev_image.astype(dtype=np.int))
        for i in range(len(possible_stain_pixels), -1, -1):
            x, y = possible_stain_pixels[i]
            if np.sum(difference[y, x, :]) > tol:
                possible_stain_pixels.pop(i)

        # Update previous and current images.
        prev_image = current_image
        ret, current_image = video.read()
        progress.update(1)

    # Make stains from points.
    pixel_cluster_list = make_pixel_clusters(pixels=possible_stain_pixels)
    pixel_cluster_list = [
        [
            (
                FramePoint(x=x/w, y=y/h),
                RGBPoint(r=prev_image[y, x, 2],
                         g=prev_image[y, x, 1],
                         b=prev_image[y, x, 0])
            )
            for x, y in cluster
        ]
        for cluster in pixel_cluster_list
    ]
    stain_list = [Stain.from_pixels(pixels=pixel_cluster) for pixel_cluster in pixel_cluster_list]
    progress.close()

    return stain_list


def make_pixel_clusters(pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    pixels = pixels[:]
    clusters: List[List[Tuple[int, int]]] = []
    current_cluster: List[Tuple[int, int]] = []
    while len(pixels) > 0:
        if len(current_cluster) == 0:
            current_cluster.append(pixels.pop())
            continue

        # Get pixels neighbour to current cluster.
        neighbour_pixels = []
        for x, y in current_cluster:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue

                    # Neighbour pixels do not belong to the current cluster.
                    new_x = x + i
                    new_y = y + j
                    already_in_cluster = False
                    for x1, y1 in current_cluster:
                        if x1 == new_x and y1 == new_y:
                            already_in_cluster = True
                            break

                    if already_in_cluster:
                        continue

                    # Neighbour pixels belong to the big list of pixels.
                    pixel_index: Optional[int] = None
                    for k, (x1, y1) in enumerate(pixels):
                        if x1 == new_x and y1 == new_y:
                            pixel_index = k
                            break

                    if pixel_index is None:
                        continue

                    # Add neighbour.
                    neighbour_pixels.append(pixels.pop(pixel_index))

        # If there are no neighbours then cluster has ended we add it to the list of existing clusters and continue.
        if len(neighbour_pixels) == 0:
            clusters.append(current_cluster)
            current_cluster = []
        else:
            current_cluster.extend(neighbour_pixels)

    if len(current_cluster) != 0:
        clusters.append(current_cluster)

    return clusters
