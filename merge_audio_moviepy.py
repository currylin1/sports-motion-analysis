# Python 3.12 / merge_audio_moviepy.py
# 目的：把「原始影片音訊」合併進「分析後無聲影片」
# - MoviePy v2：from moviepy import VideoFileClip，使用 clip.with_audio(...)
# - MoviePy v1：from moviepy.editor import VideoFileClip，使用 clip.set_audio(...)
from __future__ import annotations
import os
# 一定要在 import moviepy 之前設定
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()          # 取封包內附的 ffmpeg 路徑
    os.environ["IMAGEIO_FFMPEG_EXE"] = ff
    from moviepy.config import change_settings
    change_settings({"FFMPEG_BINARY": ff})        # 兼容舊版 moviepy 的設定方式
except Exception:
    pass

from pathlib import Path

def _import_v2_or_v1():
    """
    先嘗試 v2 匯入方式，失敗就退回 v1。
    回傳 (VideoFileClip, api_version_str)
    """
    try:
        from moviepy import VideoFileClip, AudioFileClip  # v2 方式（editor 已被拿掉）
        return VideoFileClip, "v2"
    except Exception:
        # v1 相容
        from moviepy.editor import VideoFileClip , AudioFileClip # 只有舊版才有 editor
        return VideoFileClip, "v1"

def merge_audio_with_moviepy(orig_path: str, analyzed_silent_path: str, out_path: str):
    orig_path = str(Path(orig_path))
    analyzed_silent_path = str(Path(analyzed_silent_path))
    out_path = str(Path(out_path))

    VideoFileClip, api = _import_v2_or_v1()

    # 用 with 確保檔案被關閉
    with VideoFileClip(analyzed_silent_path) as v:
        with VideoFileClip(orig_path) as orig:
            a = orig.audio
            if a is None:
                # 原檔沒有音訊就直接輸出無聲版
                clip_to_write = v
            else:
                if api == "v2":
                    # v2：改成 with_audio（.set_* -> .with_*）
                    clip_to_write = v.with_audio(a)
                else:
                    # v1：舊 API
                    clip_to_write = v.set_audio(a)

            # 仍建議寫成 H.264 + AAC，通用度最好
            clip_to_write.write_videofile(
                out_path,
                codec="libx264",
                audio_codec="aac",
                preset="veryfast",
                threads=0,
                temp_audiofile="~temp_audio.m4a",
                remove_temp=True
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        merge_audio_with_moviepy(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("用法：python merge_audio_moviepy.py 原始.mp4 無聲分析.mp4 輸出_帶聲.mp4")
