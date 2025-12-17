# yolo_person_cut_cpu.py
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse, time

from merge_audio_moviepy import merge_audio_with_moviepy  # 用你的

def open_writer(path: Path, fps: float, size):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="輸入影片路徑")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5)
    return ap.parse_args()

def main():
    args = build_args()
    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (isinstance(fps, (int, float)) and fps > 0):
        fps = 30.0
    delay_ms = max(int(1000 / fps), 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = in_path.parent / "yolo_cpu_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{in_path.stem}_yolo_person.mp4"
    writer, real_video_path = open_writer(out_video_path, fps, (w, h))

    if not args.no_display:
        cv2.namedWindow("YOLO Person CPU", cv2.WINDOW_NORMAL)

    print("[INFO] 載入 YOLOv8n-seg (CPU)...")
    # ⭐ 強制用 CPU
    model = YOLO("yolov8n-seg.pt")  # 最小最快的 seg 模型

    fidx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame,
                imgsz=640,
                conf=args.conf,
                classes=[0],   # person 類別
                device="cpu",  # ⭐ 明確指定用 CPU
                verbose=False
            )

            mask_combined = np.zeros((h, w), dtype=np.float32)

            if results:
                r = results[0]
                masks = getattr(r, "masks", None)
                if masks is not None and getattr(masks, "data", None) is not None:
                    data = masks.data.cpu().numpy()  # (N, H, W)
                    for m in data:
                        if m.shape != (h, w):
                            m = cv2.resize(m, (w, h))
                        mask_combined = np.maximum(mask_combined, m.astype(np.float32))

            mask_binary = (mask_combined > 0.4).astype(np.float32)
            kernel = np.ones((5, 5), np.uint8)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            alpha = cv2.GaussianBlur(mask_binary, (21, 21), 0)
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

            out_frame = (frame.astype(np.float32) * alpha_3).astype(np.uint8)
            writer.write(out_frame)

            if not args.no_display:
                cv2.imshow("YOLO Person CPU", out_frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                time.sleep(delay_ms / 1000.0)

            fidx += 1
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        out_with_audio = str(real_video_path.with_name(real_video_path.stem + "_audio.mp4"))
        merge_audio_with_moviepy(str(in_path), str(real_video_path), out_with_audio)
        print(f"[OK] 已保留原始聲音：{out_with_audio}")
        print(f"[OK] 影片輸出：{real_video_path}")

if __name__ == "__main__":
    main()
