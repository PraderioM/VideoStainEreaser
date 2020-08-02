import os
from tempfile import TemporaryDirectory

import cv2

from parameters_setting import get_video_path, get_fps, get_stains_path, get_out_stains_path, get_out_video_path
from remove_stains import remove_stains
from stain_selection.io import get_stored_stains, store_stains
from stain_selection.define_stains import define_stains
from copy_audio import copy_audio


def main():
    # region parameter setting.
    video_path = get_video_path()
    fps = get_fps()

    # If stains are not defined we need to define them.
    stains_path = get_stains_path()
    if stains_path is None:
        stain_list = define_stains(path=video_path)
        stains_path = get_out_stains_path(video_path=video_path)
        store_stains(stain_list=stain_list, path=stains_path)

    out_video_path = get_out_video_path(in_video_path=video_path)
    # endregion.

    # region stain removal.
    video = cv2.VideoCapture(video_path)
    stain_list = get_stored_stains(stains_path)
    with TemporaryDirectory() as tmp_dir:
        tmp_out_video_path = os.path.join(tmp_dir, os.path.basename(out_video_path))
        tmp_audio_path = os.path.join(tmp_dir, 'audio.mp3')
        remove_stains(video=video, stain_list=stain_list, out_path=tmp_out_video_path, fps=fps)
        copy_audio(in_video_images_path=tmp_out_video_path,
                   in_video_audio_path=video_path,
                   generated_audio_path=tmp_audio_path,
                   out_video_path=out_video_path)
    # endregion.


if __name__ == '__main__':
    main()
