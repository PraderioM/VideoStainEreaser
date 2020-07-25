from typing import List, Optional

import cv2
from tqdm import tqdm

from models import Stain


def remove_stains(video: cv2.VideoCapture, stain_list: List[Stain], out_path: str, fps: Optional[float]):
    # Read first frame.
    if fps is None:
        fps = video.get(cv2.CAP_PROP_FPS)
    ret, img = video.read()
    h, w, d = img.shape

    # Get video writer.
    video_writer = cv2.VideoWriter()
    video_writer.open(filename=out_path,
                      fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      fps=fps,
                      frameSize=(w, h),
                      isColor=True)

    region_pixels_list = [stain.get_region_pixels(h=h, w=w) for stain in stain_list]
    stain_pixels_list = [
        stain.get_stain_pixels(image=img, region_pixels=region_pixels)
        for stain, region_pixels in zip(stain_list, region_pixels_list)
    ]
    averaging_pixels_list = [
        stain.get_averaging_pixels(image=img, region_pixels=region_pixels, stain_pixels=stain_pixels)
        for stain, region_pixels, stain_pixels in zip(stain_list, region_pixels_list, stain_pixels_list)
    ]

    progress = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    while ret:
        # Fill stains with mean value of surrounding pixels.
        out_img = img.copy()
        for stain, stain_pixels, averaging_pixels in zip(stain_list, stain_pixels_list, averaging_pixels_list):
            out_img = stain.fill(image=out_img, stain_pixels=stain_pixels, averaging_pixels=averaging_pixels)

        # Store results.
        video_writer.write(out_img)
        progress.update(1)

        # Get next image.
        ret, img = video.read()

    # Close progressbar.
    progress.close()

    # release video writer.
    video_writer.release()
