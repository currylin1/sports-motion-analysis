# person_fg_only_opencv.py
# Python 3.12 / OpenCV-only
# 功能：對一個現成的影片檔做「前景（人）保留、背景變黑」

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def build_args():
    ap = argparse.ArgumentParser(
        description="使用 OpenCV MOG2 做前景保留（背景移除）"
    )
    ap.add_argument(
        "--video", required=True,
        help="要處理的影片路徑（建議先用 play_with_axes 產出的分析影片）"
    )
    ap.add_argument(
        "--out", default=None,
        help="輸出影片路徑（預設為 原檔名_add_fg_only.mp4）"
    )
    ap.add_argument(
        "--warmup", type=int, default=50,
        help="前幾幀只拿來訓練背景，不輸出（固定鏡頭時可以多一點）"
    )
    ap.add_argument(
        "--history", type=int, default=500,
        help="背景模型歷史長度（MOG2 參數）"
    )
    ap.add_argument(
        "--varThreshold", type=float, default=16.0,
        help="MOG2 的 varThreshold（越大越不敏感）"
    )
    ap.add_argument(
        "--no-shadows", dest="detect_shadows",
        action="store_false", default=True,
        help="關閉陰影偵測（預設開啟）"
    )
    ap.add_argument(
        "--no-display", action="store_true",
        help="不開啟視窗預覽（適合從 GUI 或背景批次呼叫）"
    )
    return ap.parse_args()


def open_writer(path: Path, fps: float, size):
    """
    與你 play_with_axes 類似的開檔邏輯：
    - 先試 mp4v
    - 不行就用 XVID + .avi 避免 0KB
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not out.isOpened():
        alt = path.with_suffix(".avi")
        out = cv2.VideoWriter(
            str(alt),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            size
        )
        if not out.isOpened():
            raise RuntimeError("VideoWriter 開啟失敗：沒有可用編碼器（可能缺少 ffmpeg 或 H.264 編碼）")
        print(f"[WARN] mp4 失敗，改存 {alt}")
        return out, alt
    return out, path


def main():
    args = build_args()

    in_path = Path(args.video)
    if not in_path.exists():
        print(f"[ERROR] 找不到影片：{in_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片：{in_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (isinstance(fps, (int, float)) and fps > 0):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)

    print(f"[INFO] 解析度: {w}x{h}, FPS: {fps:.2f}, 幀數: {total if total > 0 else '未知'}")

    # 預設輸出檔名
    if args.out is None:
        out_path = in_path.with_name(in_path.stem + "_fg_only.mp4")
    else:
        out_path = Path(args.out)

    writer, real_out_path = open_writer(out_path, fps, (w, h))

    # 建立背景減除器
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.varThreshold,
        detectShadows=args.detect_shadows,
    )

    frame_idx = 0
    delay_ms = max(int(1000 / fps), 1)

    if not args.no_display:
        cv2.namedWindow("FG Only", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 取得 MOG2 原始 mask
            # detectShadows=True 時，陰影像素值約 127
            raw_mask = bg_sub.apply(frame)

            # 暖機階段只訓練背景，不輸出
            if frame_idx < args.warmup:
                frame_idx += 1
                continue

            # 二值化：將陰影排除，只保留「極亮」的前景
            if args.detect_shadows:
                _, fgmask = cv2.threshold(raw_mask, 240, 255, cv2.THRESH_BINARY)
            else:
                _, fgmask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

            # 形態學處理：去雜訊、補掉小洞
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

            # 柔化邊界，減少鋸齒感（11x11 高斯模糊）
            fgmask_blur = cv2.GaussianBlur(fgmask, (11, 11), 0)

            # 將單通道 mask 轉成 [0,1] 浮點，並擴展成 3 通道
            alpha = fgmask_blur.astype(np.float32) / 255.0
            alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

            # 前景 = frame * alpha；背景 = 黑色 * (1-alpha)
            # 這裡背景直接用黑色，可以改成自訂顏色或模糊背景
            foreground = (frame.astype(np.float32) * alpha_3).astype(np.uint8)

            writer.write(foreground)

            if not args.no_display:
                cv2.imshow("FG Only", foreground)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        print(f"[OK] 影片輸出：{real_out_path}")


if __name__ == "__main__":
    main()
