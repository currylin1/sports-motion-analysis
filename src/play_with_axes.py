# Python 3.12
# 讀影片 -> MediaPipe Pose -> 計算角度（不畫線）+ 背景移除（YOLO）
# --remove-bg 透過 YOLOv8 分割 + 骨架遮罩，只保留人物

from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "Agg"   # 強制用無 GUI 的 backend

import cv2
from pathlib import Path
import sys, csv, math, argparse, time
from datetime import timedelta

import numpy as np  # 人像遮罩 & alpha 混合

from pose_axes import (
    PoseEstimator,
    draw_axes_on_frame,              # 目前不使用，但保留匯入
    draw_joint_pairs_and_angles,
    draw_compass_overlay,           # 目前不使用，但保留匯入
)
from merge_audio_moviepy import merge_audio_with_moviepy

# YOLOv8 segmentation
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None   # 沒裝 ultralytics 時做 graceful fallback


def RUNTIME_BASE() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))


OUTDIR = RUNTIME_BASE() / "輸出影片"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 讓 moviepy 使用隨包的 ffmpeg（避免打包後找不到）
try:
    import imageio_ffmpeg
    from moviepy.config import change_settings
    change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
except Exception:
    pass


def OUTPUT_BASE() -> Path:
    """開發時回傳原始碼路徑；封包後回傳 exe 所在目錄。"""
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


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


# ---------------------------------------------------------
# 參數設定
# ---------------------------------------------------------

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="?", default=None, help="影片路徑")
    ap.add_argument("--axis", choices=["none", "limb", "full", "fixed"], default="none")
    ap.add_argument("--fixed_len", type=float, default=220.0)
    # 羅盤參數保留但不再實際繪製
    ap.add_argument("--compass", action="store_true", default=True)
    ap.add_argument("--no-compass", dest="compass", action="store_false")
    ap.add_argument("--prefer", choices=["right_leg", "left_leg"], default="right_leg")
    # 給 GUI 使用：不開啟 OpenCV 視窗
    ap.add_argument("--no-display", action="store_true", help="不開啟 cv2.imshow 視窗（只輸出影片檔）")
    # 控制是否啟用背景移除
    ap.add_argument("--remove-bg", action="store_true", help="使用 YOLO 分割，只保留人物像素")
    return ap.parse_args()


# ---------------------------------------------------------
# 背景移除相關工具
# ---------------------------------------------------------

def build_pose_mask(kps: dict, h: int, w: int, vis_thresh: float = 0.5) -> np.ndarray:
    """
    用 Pose 關節畫出一個「極瘦」的人型遮罩，只保留身體核心區域。
    - kps: PoseEstimator.infer 回傳的 dict: name -> (x, y, visibility)
    - 回傳: (h, w) 的 float32，值在 0~1 之間
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(kps, dict):
        return mask.astype(np.float32) / 255.0

    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    base = min(h, w)
    # 很瘦的骨架：盡量不要撐出輪廓
    thickness = max(int(base * 0.015), 4)
    joint_radius = max(int(base * 0.02), 6)

    def get_pt(name):
        if name not in kps:
            return None
        v = kps[name]
        if v is None:
            return None
        if len(v) >= 3:
            x, y, vis = v[:3]
        else:
            x, y = v[:2]
            vis = 1.0
        if vis < vis_thresh:
            return None
        return int(x), int(y)

    # 骨架線
    for a, b in edges:
        pa = get_pt(a)
        pb = get_pt(b)
        if pa is not None and pb is not None:
            cv2.line(mask, pa, pb, 255, thickness=thickness)

    # 關節圓
    for name, v in kps.items():
        if v is None:
            continue
        if len(v) >= 6:
            x, y, vis = v[:3]
        else:
            x, y = v[:2]
            vis = 1.0
        if vis < vis_thresh:
            continue
        cv2.circle(mask, (int(x), int(y)), joint_radius, 255, thickness=-1)

    return mask.astype(np.float32) / 255.0


def build_body_hull_mask(kps: dict, h: int, w: int, vis_thresh: float = 0.5) -> np.ndarray:
    """
    使用所有可見關節點 + 人工頭頂點做 convex hull，生成一整塊「身體輪廓」遮罩。
    會特別把頭部（nose / 耳朵）納入，避免凸包把頭切掉。
    """
    if not isinstance(kps, dict):
        return np.zeros((h, w), dtype=np.float32)

    pts = []

    def add_pt(name: str, min_vis: float = 0.2, force: bool = False):
        """收集一個關節點；force=True 時放寬 visibility 門檻。"""
        if name not in kps:
            return
        v = kps[name]
        if v is None:
            return
        if len(v) >= 3:
            x, y, vis = v[:3]
        else:
            x, y = v[:2]
            vis = 1.0
        if not force and vis < min_vis:
            return
        pts.append([int(x), int(y)])

    # 一般身體關節：用較高 visibility 門檻
    for name, v in kps.items():
        if v is None:
            continue
        if len(v) >= 3:
            x, y, vis = v[:3]
        else:
            x, y = v[:2]
            vis = 1.0
        if vis < vis_thresh:
            continue
        pts.append([int(x), int(y)])

    # 頭部關節：nose / 左右耳，放寬 visibility 要求，強制納入
    add_pt("nose", min_vis=0.15, force=True)
    add_pt("left_ear", min_vis=0.15, force=True)
    add_pt("right_ear", min_vis=0.15, force=True)

    if len(pts) < 3:
        # 點太少，無法做 hull
        return np.zeros((h, w), dtype=np.float32)

    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    # 嘗試建立一個「頭頂」的虛擬點：用 nose 與兩肩推算
    nose = kps.get("nose")
    l_sh = kps.get("left_shoulder")
    r_sh = kps.get("right_shoulder")

    head_top_pt = None
    if nose is not None and l_sh is not None and r_sh is not None:
        if len(l_sh) >= 2 and len(r_sh) >= 2 and len(nose) >= 2:
            sx = (l_sh[0] + r_sh[0]) * 0.5
            sy = (l_sh[1] + r_sh[1]) * 0.5
            nx, ny = nose[0], nose[1]
            dx = nx - sx
            dy = ny - sy
            hx = nx + dx
            hy = ny + dy
            hx = int(np.clip(hx, 0, w - 1))
            hy = int(np.clip(hy, 0, h - 1))
            head_top_pt = [hx, hy]

    if head_top_pt is not None:
        pts_arr = np.vstack(
            [pts_arr, np.array([[head_top_pt]], dtype=np.int32)]
        )

    # 做凸包
    hull = cv2.convexHull(pts_arr)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # 稍微膨脹一點，避免把身體切太貼
    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, k, iterations=2)   # ★ 原本 1，改成 2：讓 hull 稍微變大一圈

    return mask.astype(np.float32) / 255.0


def apply_person_segmentation_yolo(
    frame_bgr: np.ndarray,
    kps: dict,
    yolo_model,
    conf: float = 0.5,
    seg_thresh: float = 1.5,   # 這個變成「近身區」的門檻
    vis_thresh: float = 0.5,
) -> np.ndarray:
    """
    YOLO segmentation + 骨架距離分區 + body hull：

    - 先跑 YOLO 得到連續機率 yolo_prob (0~1)。
    - 用 Pose 畫出「骨架細線」，向外膨脹一圈，當作 near_body 區域。
    - near_body 區域：用低門檻 (low_thr) 判斷是否保留（保人）。
    - 其他 far 區域：用高門檻 (high_thr) 判斷（砍背景）。
    - 再和 body_hull 做聯集補洞，最後模糊+雙門檻壓縮。
    """
    h, w = frame_bgr.shape[:2]

    # ---------- YOLO 分割 ----------
    if yolo_model is None:
        # 沒 YOLO 就退回純 hull
        combined = build_body_hull_mask(kps, h, w, vis_thresh=vis_thresh)
    else:
        results = yolo_model.predict(
            frame_bgr,
            imgsz=640,
            conf=conf,
            classes=[0],   # person
            device="cpu",
            verbose=False,
        )

        yolo_prob = np.zeros((h, w), dtype=np.float32)

        if results:
            r = results[0]
            masks = getattr(r, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None:
                data = masks.data.cpu().numpy()  # (N, H, W)
                if data.shape[0] > 0:
                    # 只取面積最大的那一個人
                    areas = data.reshape(data.shape[0], -1).sum(axis=1)
                    idx = int(np.argmax(areas))
                    m = data[idx]
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h))
                    yolo_prob = m.astype(np.float32)

        # ---------- 依「離骨架距離」決定門檻 ----------
        # 1) 畫一個很瘦的骨架遮罩
        pose_core = build_pose_mask(kps, h, w, vis_thresh=vis_thresh)  # 0~1

        # 2) 向外膨脹，得到「靠身體很近」的區域
        k_near = np.ones((21, 21), np.uint8)  # 視解析度，可再調
        near = cv2.dilate((pose_core > 0).astype(np.uint8), k_near, iterations=2)
        near = near.astype(np.float32)
        far = 1.0 - near

        # 門檻：近身區較寬鬆、遠區較嚴格
        low_thr = seg_thresh        # 例如 0.45
        high_thr = 0.95             # 遠離身體要很有把握才留

        mask_near = ((yolo_prob >= low_thr).astype(np.float32) * near)
        mask_far = ((yolo_prob >= high_thr).astype(np.float32) * far)

        yolo_bin = np.maximum(mask_near, mask_far)

        # 去除小雜點
        k3 = np.ones((3, 3), np.uint8)
        yolo_bin = cv2.morphologyEx(yolo_bin, cv2.MORPH_OPEN, k3, iterations=1)

        # ---------- body hull 聯集補洞 ----------
        body_hull = build_body_hull_mask(kps, h, w, vis_thresh=vis_thresh)

        area_yolo = float(yolo_bin.sum())
        if area_yolo < 50.0:
            # YOLO 幾乎沒抓到：退回 hull
            combined = body_hull
        else:
            # 聯集：YOLO + hull
            combined = np.maximum(yolo_bin, body_hull)

    # ---------- 邊緣處理 ----------
    # 補洞、讓輪廓連續
    k5 = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5, iterations=1)

    # 模糊邊緣
    alpha = cv2.GaussianBlur(combined, (7, 7), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 雙門檻壓縮
    low, high = 0.25, 0.90
    alpha = (alpha - low) / max(high - low, 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 砍掉超淡灰霧
    alpha[alpha < 0.05] = 0.0

    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)
    out = (frame_bgr.astype(np.float32) * alpha_3).astype(np.uint8)
    return out



# ---------------------------------------------------------
# 主程式
# ---------------------------------------------------------

def main():
    args = build_args()
    base = OUTPUT_BASE()
    in_path = Path(args.video) if args.video else (base / "測試影片" / "780421769.061604.mp4")
    if not in_path.exists():
        raise FileNotFoundError(f"找不到影片：{in_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (isinstance(fps, (int, float)) and fps > 0):
        fps = 30.0
    delay_ms = max(int(1000 / fps), 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
    print(f"[INFO] 解析度: {w}x{h}, FPS: {fps:.2f}, 幀數: {total if total > 0 else '未知'}")

    out_dir = base / "輸出影片"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{in_path.stem}_with_compass.mp4"
    csv_path = out_dir / f"{in_path.stem}_with_compass.csv"  # 檔名沿用舊版，避免 GUI 需要改
    writer, real_video_path = open_writer(out_video_path, fps, (w, h))
    csv_fp = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow([
        "frame", "time", "left_elbow_deg", "right_elbow_deg", "left_knee_deg", "right_knee_deg",
    ])

    # Pose
    try:
        pose = PoseEstimator(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except ImportError as e:
        print("[ERROR] MediaPipe 載入失敗：", e)
        sys.exit(1)

    # YOLOv8 segmentation 模型（只有在 remove-bg 時才載入）
    yolo_model = None
    if args.remove_bg:
        if YOLO is None:
            print("[WARN] 未安裝 ultralytics/YOLO，無法移除背景，將直接使用原圖")
        else:
            print("[INFO] 載入 YOLOv8n-seg 模型（CPU）...")
            yolo_model = YOLO("yolov8n-seg.pt")   # 第一次會自動下載權重

    # 只有在非 headless 才建立視窗
    if not args.no_display:
        cv2.namedWindow("Compass", cv2.WINDOW_NORMAL)

    # ---------------------------------------------------------
    # 主迴圈
    # ---------------------------------------------------------
    fidx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            base_frame = frame.copy()

            # 1) Pose 推論（用原畫面）
            kps = pose.infer(base_frame)

            # 2) 先做背景移除（如果啟用）
            if args.remove_bg:
                vis_frame = apply_person_segmentation_yolo(
                    base_frame,
                    kps,
                    yolo_model=yolo_model,
                    conf=0.5,
                    seg_thresh=0.45,
                    vis_thresh=0.5,
                )
            else:
                vis_frame = base_frame

            # 3) 計算關節與角度，但不在畫面上畫線條
            tmp_canvas = np.zeros_like(vis_frame)
            joints = draw_joint_pairs_and_angles(
                tmp_canvas,  # 只拿來算角度，不用來顯示
                kps,
                vis_thresh=0.5,
                draw_angles=False,
            )

            # 軸線 / 羅盤全部取消（畫面只剩去背後的人像）

            # 4) 寫 CSV
            def get_angle(d, k):
                return (d[k].angle_deg if (k in d and d[k] is not None) else float("nan"))

            t = str(timedelta(seconds=fidx / fps))
            row = [
                fidx, t,
                get_angle(joints, "left_elbow"),
                get_angle(joints, "right_elbow"),
                get_angle(joints, "left_knee"),
                get_angle(joints, "right_knee"),
            ]
            csv_writer.writerow(row)

            # 5) 寫出影片
            writer.write(vis_frame)

            # 6) 顯示（非 headless 才顯示）
            if not args.no_display:
                cv2.imshow("Compass", vis_frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == 32:
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 in (32, 27, ord('q')):
                            break
                    if k2 in (27, ord('q')):
                        break
            else:
                time.sleep(delay_ms / 1000.0)

            fidx += 1
    finally:
        cap.release()
        writer.release()
        csv_fp.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        # 併回原始音訊
        out_with_audio = str(real_video_path.with_name(real_video_path.stem + "_audio.mp4"))
        merge_audio_with_moviepy(str(in_path), str(real_video_path), out_with_audio)
        print(f"[OK] 已保留原始聲音：{out_with_audio}")
        print(f"[OK] 影片輸出：{real_video_path}")
        print(f"[OK] CSV 輸出：{csv_path}")


if __name__ == "__main__":
    main()
