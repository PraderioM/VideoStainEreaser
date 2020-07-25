import os
from typing import Optional

import cv2

NO_ANSWERS = ('n', 'no', 'nope')
YES_ANSWERS = ('y', 'yes', 'yep')


def get_video_path():
    while True:
        video_path = input('Insert path to video:\n\t')
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            print(f'There is no video with location `{video_path}`.')
        else:
            video_capture = cv2.VideoCapture(video_path)
            ret, _ = video_capture.read(video_path)
            if ret:
                return video_path
            else:
                print(f'Could not read video located in `{video_path}`.')

        print('Please insert a valid path.')


def get_fps() -> Optional[float]:
    while True:
        fps_string = input('Insert video fps. If nothing is specified fps will be inferred from video metadata but video metadata may be wrong:\n\t')

        if len(fps_string) == 0:
            return None

        try:
            return float(fps_string)
        except ValueError:
            print('Please insert a valid number either an integer or float.')


def get_stains_path() -> Optional[str]:
    while True:
        stains_path = input('Insert path to json file containing stains information. If no such file exists of you want to create it anew leave this region black:\n\t')

        if len(stains_path) == 0:
            return None

        stains_path = os.path.abspath(stains_path)
        if not os.path.exists(stains_path):
            print(f'There is no file with location `{stains_path}`.')
        else:
            return stains_path

        print('Please insert a valid path.')


def get_out_stains_path() -> str:
    while True:
        stains_path = input('Insert path to json file where stains will be stored:\n\t')
        stains_path = os.path.abspath(stains_path)

        if not os.path.exists(os.path.dirname(stains_path)):
            out_path = _optionally_create_directory(stains_path)
            if out_path is None:
                continue
            else:
                return out_path

        print('Please insert a valid path.')


def get_out_video_path() -> str:
    while True:
        out_path = input('Insert output video path:\n\t')
        out_path = os.path.abspath(out_path)
        if os.path.exists(out_path):
            while True:
                proceed = input(f'There already exists a video with location `{out_path}`. Are you sure you want to proceed? If you do previous video will be removed: [y/N]')
                if proceed.lower() in list(NO_ANSWERS) + ['']:
                    break
                elif proceed.lower() in YES_ANSWERS:
                    return out_path
                else:
                    print(f"I don't understand your answer. Please repeat it.")
            continue
        else:
            out_path = _optionally_create_directory(out_path)
            if out_path is None:
                continue
            else:
                return out_path


def get_size(size_name: str, default_val: int = 512) -> int:
    while True:
        size_string = input(f'Insert {size_name}: [{default_val}]\n\t')

        if len(size_string) == 0:
            return default_val

        if not size_string.isnumeric():
            print('Please insert a valid integer.')
        else:
            return int(size_string)


def get_define_stains_manually() -> bool:
    while True:
        define_manually = input(f'Doy you wish to define stains manually?: [y/N]')
        if define_manually.lower() in list(NO_ANSWERS) + ['']:
            return False
        elif define_manually.lower() in YES_ANSWERS:
            return True
        else:
            print(f"I don't understand your answer. Please repeat it.")


def _optionally_create_directory(path: str) -> Optional[str]:
    if not os.path.exists(os.path.dirname(path)):
        while True:
            proceed = input(
                f'There exists no directory named `{os.path.dirname(path)}`. Do you want me to create it?: [Y/n]')
            if proceed.lower() in NO_ANSWERS:
                break
            elif proceed.lower() in list(YES_ANSWERS) + ['']:
                os.makedirs(os.path.dirname(path))
                return path
            else:
                print(f"I don't understand your answer. Please repeat it.")
        return None
    else:
        return path
