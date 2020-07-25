from typing import List, Optional, Tuple

import cv2
import numpy as np

from models.stain import Stain
from models.frame_region import FramePoint, FrameRegion
from models.rgb_region import RGBPoint, RGBRegion
from parameters_setting import get_size
from stain_selection.constants import VIDEO_WINDOW_NAME, PALETTE_WINDOW_NAME


class RefreshIndicator:
    def __init__(self, refresh_video: bool = True, refresh_color_palette: bool = True):
        self.refresh_video = refresh_video
        self.refresh_color_palette = refresh_color_palette


def define_stains(path: str) -> List[Stain]:
    print_instructions()

    video = cv2.VideoCapture(path)
    show_h = get_size(size_name='showed image height')
    show_w = get_size(size_name='showed image width')

    create_windows()
    stain_list: List[Optional[Stain]] = [None]
    refresh_indicator = RefreshIndicator()
    set_mouse_callbacks(stain_list=stain_list, refresh_indicator=refresh_indicator, show_h=show_h, show_w=show_w)
    start_loop(video=video, stain_list=stain_list, refresh_indicator=refresh_indicator, show_h=show_h, show_w=show_w)

    return [stain for stain in stain_list if stain is not None]


def create_windows():
    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(PALETTE_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)


def set_mouse_callbacks(stain_list: List[Optional[Stain]], refresh_indicator: RefreshIndicator,
                        show_h: int = 512, show_w: int = 512):
    # Video mouse callbacks.
    def video_mouse_callback(event: int, x: int, y: int, flags: int, param: Optional):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(stain_list) == 0 or stain_list[-1] is None:
                stain_list.pop()
                new_frame_region = FrameRegion(points=[FramePoint(x=x/show_w, y=y/show_h),
                                                       FramePoint(x=x/show_w, y=y/show_h)])
                new_rgb_region = RGBRegion([])
                new_stain = Stain(frame_region=new_frame_region, rgb_region=new_rgb_region)
                stain_list.append(new_stain)
            else:
                frame_region = stain_list[-1].frame_region
                frame_region.add_point(FramePoint(x=x/show_w, y=y/show_h))
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected_point = FramePoint(x=x / show_w, y=y / show_h)
            i = 0
            while i < len(stain_list):
                stain = stain_list[i]
                if stain is None:
                    i += 1
                    continue

                frame_region = stain.frame_region

                if frame_region.contains_point(x=selected_point.x, y=selected_point.y):
                    stain_list.pop(i)
                else:
                    i += 1
        elif len(stain_list) != 0 and stain_list[-1] is not None:
            frame_region = stain_list[-1].frame_region
            frame_region.replace_last_point(point=FramePoint(x=x/show_w, y=y/show_h))

    cv2.setMouseCallback(VIDEO_WINDOW_NAME, video_mouse_callback)

    # Palette mouse callbacks
    def palette_mouse_callback(event: int, x: int, y: int, flags: int, param: Optional):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(stain_list) == 0 or stain_list[-1] is None:
            return

        rgb_region = stain_list[-1].rgb_region
        if x < rgb_region.size and y < rgb_region.size:
            color = {
                rgb_region.x_axis: min(255, int(256 * x / rgb_region.size)),
                rgb_region.y_axis: max(0, 255 - int(256 * y / rgb_region.size)),
                rgb_region.z_axis: rgb_region.z_axis_val,
            }
            rgb_region.add_point(RGBPoint(r=color['r'], g=color['g'], b=color['b']))
            refresh_indicator.refresh_color_palette = True
        elif x < rgb_region.size and y >= rgb_region.size + int(rgb_region.z_bar_height / 2):
            rgb_region.z_axis_val = min(255, int(256 * x / rgb_region.size))
            refresh_indicator.refresh_color_palette = True
        elif x < rgb_region.size / 2 and rgb_region.size <= y < rgb_region.size + rgb_region.z_bar_height / 2:
            rgb_region.change_x_axis()
            refresh_indicator.refresh_color_palette = True
        elif x >= rgb_region.size / 2 and rgb_region.size <= y < rgb_region.size + rgb_region.z_bar_height / 2:
            rgb_region.change_y_axis()
            refresh_indicator.refresh_color_palette = True
        elif x >= rgb_region.size and y >= rgb_region.size:
            rgb_region.clear_region()
            refresh_indicator.refresh_color_palette = True

    cv2.setMouseCallback(PALETTE_WINDOW_NAME, palette_mouse_callback)


def start_loop(video: cv2.VideoCapture, stain_list: List[Stain],
               refresh_indicator: RefreshIndicator,
               show_h: int = 512, show_w: int = 512):
    # Initialize images.
    ret, original_image = video.read()
    original_image = cv2.resize(original_image, dsize=(show_w, show_h))
    image = original_image.copy()
    empty_palette_image = np.zeros(shape=(show_h, show_w, 3), dtype=image.dtype)
    palette_image = empty_palette_image.copy()

    # Show images in a loop.
    while ret:
        while True:
            # Refresh video image if needed.
            if refresh_indicator.refresh_video:
                image = original_image.copy()
                for stain in stain_list:
                    if stain is None:
                        continue
                    image = stain.frame_region.draw_on_image(image=image)
                refresh_indicator.refresh_video = False

            # Refresh palette image if needed.
            if refresh_indicator.refresh_color_palette:
                if len(stain_list) == 0 or stain_list[-1] is None:
                    palette_image = empty_palette_image.copy()
                refresh_indicator.refresh_color_palette = False

            # Show images and process key-board callbacks.
            cv2.imshow(VIDEO_WINDOW_NAME, image)
            cv2.imshow(PALETTE_WINDOW_NAME, palette_image)
            key = cv2.waitKey(1) & 0xFF

            # Process pressed key.
            next_image, break_loop = process_key(key=key, stain_list=stain_list,
                                                 refresh_indicator=refresh_indicator)

            if next_image or break_loop:
                break

        if break_loop:
            break

        ret, original_image = video.read()
        original_image = cv2.resize(original_image, dsize=(show_w, show_h))
        refresh_indicator.refresh_video = True


def process_key(key: int, stain_list: List[Optional[Stain]], refresh_indicator: RefreshIndicator) -> Tuple[bool, bool]:
    next_image, break_loop = False, False
    if key == ord('s'):
        refresh_indicator.refresh_color_palette = True
        stain_list.append(None)
    elif key == ord('n'):
        refresh_indicator.refresh_video = True
        next_image = True
    elif key == ord('q'):
        break_loop = True

    return next_image, break_loop


def print_instructions():
    print("""
KEYBOARD COMMANDS:
  `s`: saves current stain and opens a new one.
  `n`: moves to the next video image.
  `q`: saves current stain and quits stain selection.
VIDEO WINDOW:
  `click`: add new point to stain region.
  `right-click`: remove existing stain.
PALETTE WINDOW:
  `click palette`: Add new vertex to region of stain color.
  `click z axis bar`: Changes z axis value of color palette.
  `click change z axis`: Places z axis in place of the selected axis.
""")
