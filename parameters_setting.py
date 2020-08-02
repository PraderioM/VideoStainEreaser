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
            ret, _ = video_capture.read()
            if ret:
                return video_path
            else:
                print(f'Could not read video located in `{video_path}`.')

        print('Please insert a valid path.')


def get_fps() -> Optional[float]:
    while True:
        fps_string = input('Insert video fps. If nothing is specified fps will be inferred from video metadata:\n\t')

        if len(fps_string) == 0:
            return None

        try:
            return float(fps_string)
        except ValueError:
            print('Please insert a valid number either an integer or float.')


def get_stains_path() -> Optional[str]:
    while True:
        stains_path = input('Insert path to json file containing stains information.\n'
                            'If you want/need to create it anew leave this region black:\n\t')

        if len(stains_path) == 0:
            return None

        stains_path = os.path.abspath(stains_path)
        if not os.path.exists(stains_path):
            print(f'There is no file with location `{stains_path}`.')
        else:
            return stains_path

        print('Please insert a valid path.')


def get_out_stains_path(video_path: Optional[str] = None) -> str:
    # Get default.
    if video_path is None:
        default_stains_path = ''
    else:
        split_video_path = list(os.path.splitext(video_path))
        if len(split_video_path) == 0:
            default_stains_path = ''
        else:
            split_video_path[-1] = 'json'
            default_stains_path = '.'.join(split_video_path)

    # Ask user for input.
    while True:
        if len(default_stains_path) == 0:
            stains_path = input('Insert path to json file where stains will be stored:\n\t')
        else:
            stains_path = input(f'Insert path to json file where stains will be stored: [{default_stains_path}]\n\t')
        if len(stains_path) == 0:
            stains_path = default_stains_path
        stains_path = os.path.abspath(stains_path)

        out_path = _optionally_create_directory(stains_path)
        if out_path is None:
            print('Please insert a valid path.')
            continue
        elif os.path.exists(out_path):
            while True:
                proceed = input(f'There already exists a file named `{out_path}`. Do you want to overwrite it?: [y/N]')
                if proceed.lower() in list(NO_ANSWERS) + ['']:
                    break
                elif proceed.lower() in YES_ANSWERS:
                    return out_path
                else:
                    print(f"I don't understand your answer. Please repeat it.")
        else:
            return out_path


def get_out_video_path(in_video_path: Optional[str] = None) -> str:
    # Get default.
    if in_video_path is None:
        default_out_path = ''
    else:
        split_video_path = list(os.path.splitext(in_video_path))
        if len(split_video_path) <= 1:
            default_out_path = ''
        else:
            split_video_path[-1] = 'avi'
            split_video_path[-2] = split_video_path[-2] + '_cleaned'
            default_out_path = '.'.join(split_video_path)

    # Ask user for input.
    while True:
        if len(default_out_path) == 0:
            out_path = input('Insert output video path:\n\t')
        else:
            out_path = input(f'Insert output video path: [{default_out_path}]\n\t')
        if len(out_path) == 0:
            out_path = default_out_path
        out_path = os.path.abspath(out_path)
        if os.path.exists(out_path):
            while True:
                proceed = input(f'There already exists a video with location `{out_path}`.\n'
                                f'Are you sure you want to proceed? If you do previous video will be removed: [y/N]')
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
                print('Please insert a valid path.')
                continue
            else:
                return out_path


def get_size(size_name: str, default_val: int = 768) -> int:
    while True:
        size_string = input(f'Insert {size_name}: [{default_val}]\n\t')

        if len(size_string) == 0:
            return default_val

        if not size_string.isnumeric():
            print('Please insert a valid integer.')
        else:
            return int(size_string)


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
