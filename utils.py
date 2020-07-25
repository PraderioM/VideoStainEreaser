from typing import List, Tuple

import cv2
import numpy as np


def draw_centered_text(image: np.ndarray,
                       text: str,
                       cx: float = 0.5, cy: float = 0.5,
                       font_scale: int = 1, font_face: int = 1,
                       color: Tuple[int, int, int] = (0, 0, 0),
                       thickness: int = 1):
    (x_shift, y_shift), _ = cv2.getTextSize(text, fontFace=font_face, fontScale=font_scale, thickness=thickness)
    x_shift = int(x_shift)
    y_shift = int(y_shift)
    h, w, _ = image.shape
    cx, cy = int(cx * w), int(cy * h)
    org = int(cx - x_shift / 2), int(cy + y_shift / 2)
    return cv2.putText(image, text=text, org=org, fontFace=font_face,
                       fontScale=font_scale, color=color, thickness=thickness)


def draw_polygon(image: np.ndarray, points: List, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    start_points = points
    end_points = points[1:] + points[:1]
    h, w, _ = image.shape
    for start_point, end_point in zip(start_points, end_points):
        image = cv2.line(img=image,
                         pt1=(int(start_point.x * w), int(start_point.y * h)),
                         pt2=(int(end_point.x * w), int(end_point.y * h)),
                         color=color,
                         thickness=thickness)

    return image
