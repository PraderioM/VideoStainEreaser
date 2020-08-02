import os


def copy_audio(in_video_images_path: str,
               in_video_audio_path: str,
               generated_audio_path: str,
               out_video_path: str):
    print('Extracting audio from original video.')
    os.system(f'ffmpeg -i {in_video_audio_path} {generated_audio_path}')
    print('Putting audio on resulting video.')
    os.system(f'ffmpeg -i {in_video_images_path} -i {generated_audio_path} -codec copy -shortest {out_video_path}')
