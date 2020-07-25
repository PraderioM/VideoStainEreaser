import cv2

from parameters_setting import get_video_path, get_fps, get_stains_path, get_out_stains_path, get_out_video_path
from parameters_setting import get_define_stains_manually
from remove_stains import remove_stains
from stain_selection.io import get_stored_stains, store_stains
from stain_selection.define_stains import define_stains
from stain_selection.get_stains_automatically import get_stains_automatically


def main():
    # region parameter setting.
    video_path = get_video_path()
    fps = get_fps()

    # If stains are not defined we need to define them.
    stains_path = get_stains_path()
    if stains_path is None:
        if get_define_stains_manually():
            stain_list = define_stains(path=video_path)
        else:
            stain_list = get_stains_automatically(path=video_path)
        stains_path = get_out_stains_path()
        store_stains(stain_list=stain_list, path=stains_path)

    out_video_path = get_out_video_path()
    # endregion.

    # region stain removal.
    video = cv2.VideoCapture(video_path)
    stain_list = get_stored_stains(stains_path)
    remove_stains(video=video, stain_list=stain_list, out_path=out_video_path, fps=fps)
    # endregion.


if __name__ == '__main__':
    main()
