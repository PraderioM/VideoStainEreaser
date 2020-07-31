from typing import List, Tuple

import cv2
import numpy as np


def draw_centered_text(image: np.ndarray,
                       text: str,
                       cx: float = 0.5, cy: float = 0.5,
                       font_scale: int = 1, font_face: int = 1,
                       color: Tuple[int, int, int] = (0, 0, 0),
                       thickness: int = 1) -> np.ndarray:
    (x_shift, y_shift), _ = cv2.getTextSize(text, fontFace=font_face, fontScale=font_scale, thickness=thickness)
    x_shift = int(x_shift)
    y_shift = int(y_shift)
    h, w, _ = image.shape
    cx, cy = int(cx * w), int(cy * h)
    org = int(cx - x_shift / 2), int(cy + y_shift / 2)
    return cv2.putText(image, text=text, org=org, fontFace=font_face,
                       fontScale=font_scale, color=color, thickness=thickness)


def draw_polygon(image: np.ndarray, points: List[Tuple[float, float]],
                 color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    start_points = points
    end_points = points[1:] + points[:1]
    h, w, _ = image.shape
    for (x_start, y_start), (x_end, y_end) in zip(start_points, end_points):
        image = cv2.line(img=image,
                         pt1=(int(x_start * w), int(y_start * h)),
                         pt2=(int(x_end * w), int(y_end * h)),
                         color=color,
                         thickness=thickness)

    return image


def cuts_segment(x: float, y: float, extreme_1: Tuple[float, float], extreme_2: Tuple[float, float]):
    x_1, y_1 = extreme_1
    x_2, y_2 = extreme_2

    # If segment is vertical we check if point lies exactly in the segment.
    if x_1 == x_2:
        return x_1 == x and y_1 <= y < y

    # If segment is not vertical we get the point vertical projection onto the segment.
    y_aux = (x - x_1) * (y_2 - y_1) / (x_2 - x_1) + y_1

    # The point cuts if the projection falls within the segments limits and above the point.
    return x_1 <= x < x_2 and y_aux >= y