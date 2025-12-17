# person_matting_video.py
# Python 3.12 可用版本
# 功能：
#   - 使用 MediaPipe SelfieSegmentation 對影片逐幀做人像去背
#   - 背景改為純黑，只保留人物
#   - 保留解析度 / FPS
#   - 最後併回原始聲音

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from datetime import timedelta

import cv2
import numpy as np

from merge_audio_moviepy import merge_audio_with_moviepy

try:
    import mediapipe as mp
except ImportError as e:
    print("[ERROR] mediapipe 未安裝，請先執行：pip install -U mediapipe")
    sys.exit(1)


def open_writer(path: Path, fps: float, size):
    """開啟 VideoWriter，優先 mp4v，失敗退回 avi。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not out.isOpened():
        alt = path.with_suffix(".avi")
        out = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        if not out.isOpened():
            raise RuntimeError("VideoWriter 開啟失敗：沒有可用編碼器")
        print(f"[WARN] mp4 失敗，改存 {alt.name}")
        return out, alt
    return out, path


def build_args():
    ap = argparse.ArgumentParser(
        description="使用 MediaPipe SelfieSegmentation 對影片做人像去背"
    )
    ap.add_argument("--video", required=True, help="輸入影片路徑")
    ap.add_argument(
        "--out",
        default=None,
        help="輸出影片路徑（預設：輸入影片同資料夾，檔名加 _mp_seg）",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="可選：內部推論時縮放最長邊到此像素（例如 720），可加速；輸出仍為原解析度。",
    )
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help="不併回原始聲音（預設會併回）。",
    )
    return ap.parse_args()


def seg_frame_with_mediapipe(
    frame_bgr: np.ndarray,
    seg_model,
    max_side: int | None = None,
    low: float = 0.25,
    high: float = 0.90,
) -> np.ndarray:
    """
    使用 MediaPipe SelfieSegmentation 對一張 BGR 幀做人像去背。

    - seg_model: mp.solutions.selfie_segmentation.SelfieSegmentation 物件
    - max_side: 內部運算的最長邊（縮小後再放大 alpha）；None = 用原解析度
    - low / high: 雙門檻，決定貼合程度：
        * < low  -> 一律當背景
        * > high -> 一律當前景
        * 中間   -> 線性過渡
    """
    h, w = frame_bgr.shape[:2]
    work_bgr = frame_bgr
    scale = 1.0

    # 內部縮放，加速運算
    if max_side is not None and max_side > 0:
        long_side = max(h, w)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_w = int(w * scale)
            new_h = int(h * scale)
            work_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    res = seg_model.process(work_rgb)
    seg_mask = getattr(res, "segmentation_mask", None)

    if seg_mask is None:
        # 分割失敗就直接回原圖
        return frame_bgr

    seg_mask = seg_mask.astype(np.float32)
    seg_mask = np.clip(seg_mask, 0.0, 1.0)

    # 若有縮放，把 mask 放大回原解析度
    if scale != 1.0:
        alpha = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        alpha = seg_mask

    # 先做一點模糊，讓邊緣不要鋸齒
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 雙門檻壓縮，把人盡量抓乾淨
    #   < low  -> 0
    #   > high -> 1
    #   中間   -> 線性 0~1
    low = float(low)
    high = float(high)
    if high <= low:
        high = low + 1e-3

    alpha = (alpha - low) / (high - low)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 很小的值直接砍掉，背景更乾淨
    alpha[alpha < 0.05] = 0.0

    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

    # 前景 = 原始 BGR * alpha，背景 = 黑色 * (1-alpha)
    out = (frame_bgr.astype(np.float32) * alpha_3).astype(np.uint8)
    return out


def main():
    args = build_args()

    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(f"找不到輸入影片：{in_path}")

    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = in_path.with_name(in_path.stem + "_mp_seg.mp4")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not isinstance(fps, (int, float)) or fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)

    print(f"[INFO] 解析度: {w}x{h}, FPS: {fps:.2f}, 幀數: {total if total > 0 else '未知'}")
    if args.max_side:
        print(f"[INFO] 內部分割最長邊縮放到：{args.max_side}px")

    writer, real_out_path = open_writer(out_path, fps, (w, h))

    # 建立 MediaPipe SelfieSegmentation
    mp_selfie = mp.solutions.selfie_segmentation
    seg_model = mp_selfie.SelfieSegmentation(model_selection=1)
    print("[INFO] 已啟用 MediaPipe SelfieSegmentation")

    fidx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out_frame = seg_frame_with_mediapipe(
                frame,
                seg_model,
                max_side=args.max_side,
                low=0.25,
                high=0.90,
            )
            writer.write(out_frame)

            fidx += 1
            if fidx % 30 == 0:
                elapsed = time.time() - t0
                fps_now = fidx / elapsed if elapsed > 0 else 0.0
                if total > 0:
                    prog = fidx / total * 100.0
                    print(f"[INFO] 處理中：{fidx}/{total} ({prog:5.1f}%), 約 {fps_now:.2f} FPS")
                else:
                    print(f"[INFO] 處理中：{fidx} 幀，約 {fps_now:.2f} FPS")

    finally:
        cap.release()
        writer.release()
        seg_model.close()

    print(f"[OK] 去背影片（無聲音）輸出：{real_out_path}")

    # 併回原始聲音
    if not args.no_audio:
        out_with_audio = real_out_path.with_name(real_out_path.stem + "_audio.mp4")
        merge_audio_with_moviepy(str(in_path), str(real_out_path), str(out_with_audio))
        print(f"[OK] 已保留原始聲音：{out_with_audio}")
    else:
        print("[INFO] 依照參數 --no-audio，未併回聲音。")


if __name__ == "__main__":
    main()
