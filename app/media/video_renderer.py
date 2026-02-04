from moviepy.editor import ImageSequenceClip, AudioFileClip
import uuid
import os

VIDEO_DIR = "data/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

def render_video(frames, audio_path):
    video_path = f"{VIDEO_DIR}/{uuid.uuid4()}.mp4"
    clip = ImageSequenceClip(frames, fps=1)
    audio = AudioFileClip(audio_path)
    clip.set_audio(audio).write_videofile(
        video_path,
        codec="libx264",
        audio_codec="aac"
    )
    return video_path
