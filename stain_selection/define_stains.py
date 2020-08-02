from typing import List, Optional, Tuple

import cv2
import numpy as np

from models.stain import Stain
from models.frame_region import FramePoint, FrameRegion
from parameters_setting import get_size
from stain_selection.constants import VIDEO_WINDOW_NAME, CROP_WINDOW_NAME


class RefreshIndicator:
    def __init__(self, refresh_video: bool = True,
                 refresh_crop: bool = True,
                 refresh_on_mouse_over: bool = True):
        self._refresh_video = refresh_video
        self.refresh_crop = refresh_crop
        self.refresh_on_mouse_over = refresh_on_mouse_over

    @property
    def refresh_video(self) -> bool:
        return self._refresh_video

    @refresh_video.setter
    def refresh_video(self, refresh: bool):
        self._refresh_video = refresh
        if refresh:
            self.refresh_crop = True


def define_stains(path: str) -> List[Stain]:
    # setup.
    show_h = get_size(size_name='showed image height')
    show_w = get_size(size_name='showed image width')
    crop_h = get_size(size_name='crop height', default_val=384)
    crop_w = get_size(size_name='crop width', default_val=384)
    print_instructions()

    video = cv2.VideoCapture(path)

    create_windows(show_h=show_h, show_w=show_w,
                   crop_h=crop_h, crop_w=crop_w)
    stain_list: List[Optional[Stain]] = [None]
    refresh_indicator = RefreshIndicator()
    set_mouse_callbacks(stain_list=stain_list, refresh_indicator=refresh_indicator,
                        show_h=show_h, show_w=show_w,
                        crop_h=crop_h, crop_w=crop_w)
    start_loop(video=video, stain_list=stain_list, refresh_indicator=refresh_indicator,
               show_h=show_h, show_w=show_w,
               crop_h=crop_h, crop_w=crop_w)
    close_windows()

    return [stain for stain in stain_list if stain is not None]


def create_windows(show_h: int = 512, show_w: int = 512,
                   crop_h: int = 256, crop_w: int = 256):
    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CROP_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)


def close_windows():
    cv2.destroyWindow(VIDEO_WINDOW_NAME)
    cv2.destroyWindow(CROP_WINDOW_NAME)


def set_mouse_callbacks(stain_list: List[Optional[Stain]], refresh_indicator: RefreshIndicator,
                        show_h: int = 512, show_w: int = 512,
                        crop_h: int = 256, crop_w: int = 256):
    # Video mouse callbacks.
    def video_mouse_callback(event: int, x: int, y: int, flags: int, param: Optional):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(stain_list) == 0 or stain_list[-1] is None:
                if len(stain_list) > 0:
                    stain_list.pop()
                new_neighborhood = FrameRegion(points=[FramePoint(x=x/show_w, y=y/show_h)])
                new_crop = FrameRegion(points=[])
                new_stain = Stain(neighborhood=new_neighborhood, crop=new_crop)
                stain_list.append(new_stain)
            else:
                neighborhood = stain_list[-1].neighborhood
                neighborhood.add_point(FramePoint(x=x/show_w, y=y/show_h))
            refresh_indicator.refresh_video = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected_point = FramePoint(x=x / show_w, y=y / show_h)
            i = 0
            while i < len(stain_list):
                stain = stain_list[i]
                if stain is None:
                    i += 1
                    continue

                neighborhood = stain.neighborhood

                if neighborhood.contains_point(x=selected_point.x, y=selected_point.y):
                    stain_list.pop(i)
                else:
                    i += 1
            refresh_indicator.refresh_video = True
        elif len(stain_list) != 0 and stain_list[-1] is not None and refresh_indicator.refresh_on_mouse_over:
            neighborhood = stain_list[-1].neighborhood
            new_point = FramePoint(x=x / show_w, y=y / show_h)
            if neighborhood.n_points == 1:
                neighborhood.add_point(point=new_point)
            else:
                neighborhood.replace_last_point(point=new_point)
            refresh_indicator.refresh_video = True

    cv2.setMouseCallback(VIDEO_WINDOW_NAME, video_mouse_callback)

    # Palette mouse callbacks
    def crop_mouse_callback(event: int, x: int, y: int, flags: int, param: Optional):
        if len(stain_list) == 0 or stain_list[-1] is None:
            return

        neighborhood = stain_list[-1].neighborhood
        crop = stain_list[-1].crop
        if event == cv2.EVENT_LBUTTONDOWN:
            point = FramePoint.from_roi(x=x / crop_w, y=y / crop_h,
                                        x_min=neighborhood.x_min, y_min=neighborhood.y_min,
                                        x_max=neighborhood.x_max, y_max=neighborhood.y_max)
            crop.add_point(point)
            refresh_indicator.refresh_video = True
            refresh_indicator.refresh_crop = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            point = FramePoint.from_roi(x=x / crop_w, y=y / crop_h,
                                        x_min=neighborhood.x_min, y_min=neighborhood.y_min,
                                        x_max=neighborhood.x_max, y_max=neighborhood.y_max)
            if crop.contains_point(x=point.x, y=point.y):
                crop.reset()
                refresh_indicator.refresh_video = True
                refresh_indicator.refresh_crop = True
        elif crop.n_points >= 1 and refresh_indicator.refresh_on_mouse_over:
            point = FramePoint.from_roi(x=x / crop_w, y=y / crop_h,
                                        x_min=neighborhood.x_min, y_min=neighborhood.y_min,
                                        x_max=neighborhood.x_max, y_max=neighborhood.y_max)
            if crop.n_points == 1:
                crop.add_point(point=point)
            else:
                crop.replace_last_point(point=point)
            refresh_indicator.refresh_video = True
            refresh_indicator.refresh_crop = True

    cv2.setMouseCallback(CROP_WINDOW_NAME, crop_mouse_callback)


def start_loop(video: cv2.VideoCapture, stain_list: List[Stain],
               refresh_indicator: RefreshIndicator,
               show_h: int = 512, show_w: int = 512,
               crop_h: int = 256, crop_w: int = 256):
    # Initialize images.
    ret, original_image = video.read()
    original_image = cv2.resize(original_image, dsize=(show_w, show_h))
    image = original_image.copy()
    empty_crop = np.zeros(shape=(crop_h, crop_w, 3), dtype=image.dtype)
    crop_image = empty_crop.copy()

    # Show images in a loop.
    while ret:
        while True:
            # Refresh video image if needed.
            if refresh_indicator.refresh_video:
                image = original_image.copy()
                for stain in stain_list:
                    if stain is None:
                        continue
                    image = stain.neighborhood.draw_on_image(image=image)
                    image = stain.crop.draw_on_image(image=image, color=(0, 0, 255))
                refresh_indicator.refresh_video = False

            # Refresh palette image if needed.
            if refresh_indicator.refresh_crop:
                if len(stain_list) == 0 or stain_list[-1] is None:
                    crop_image = empty_crop.copy()
                else:
                    crop_image = stain_list[-1].get_roi_crop(original_image, out_h=crop_h, out_w=crop_w)
                    crop_image = stain_list[-1].draw_on_roi(crop_image,
                                                            x_min=stain_list[-1].x_min, y_min=stain_list[-1].y_min,
                                                            x_max=stain_list[-1].x_max, y_max=stain_list[-1].y_max)
                refresh_indicator.refresh_crop = False

            # Show images and process key-board callbacks.
            cv2.imshow(VIDEO_WINDOW_NAME, image)
            cv2.imshow(CROP_WINDOW_NAME, crop_image)
            key = cv2.waitKey(1) & 0xFF

            # Process pressed key.
            next_image, break_loop = process_key(key=key, stain_list=stain_list,
                                                 refresh_indicator=refresh_indicator,
                                                 original_image=original_image)

            if next_image or break_loop:
                break

        if break_loop:
            break

        ret, original_image = video.read()
        original_image = cv2.resize(original_image, dsize=(show_w, show_h))
        refresh_indicator.refresh_video = True


def process_key(key: int, stain_list: List[Optional[Stain]],
                refresh_indicator: RefreshIndicator,
                original_image: np.ndarray) -> Tuple[bool, bool]:
    next_image, break_loop = False, False
    if key == ord('s'):
        refresh_indicator.refresh_video = True
        stain_list.append(None)
    elif key == ord('n'):
        refresh_indicator.refresh_video = True
        next_image = True
    elif key == ord('q'):
        break_loop = True
    elif key == ord('p'):
        refresh_indicator.refresh_on_mouse_over = not refresh_indicator.refresh_on_mouse_over
    elif key == ord('t') and len(stain_list) != 0 and stain_list[-1] is not None:
        stain = stain_list[-1]
        show_image = stain.fill(image=original_image.copy())
        win_name = 'removal result'
        cv2.imshow(win_name, show_image)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

    return next_image, break_loop


def print_instructions():
    print("""
KEYBOARD COMMANDS:
  `s`: Saves current stain and opens a new one.
  `p`: pauses/resumes refreshing of images on mouse over.
  `n`: Moves to the next video image.
  `q`: Saves current stain and quits stain selection.
  `t`: Uses defined stain in order to manipulate the image and show the result of such manipulation.
       Press any key to proceed.
VIDEO WINDOW:
  `click`: Add new vertex to stain region.
  `right-click`: Remove existing stain.
STAIN WINDOW:
  `click`: Add new vertex to precise stain region.
  `right-click`: Remove existing stain.
""")
