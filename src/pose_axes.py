# Python 3.12
# 功能總覽（保留原有功能，僅新增「羅盤/指南針式角度顯示」）：
# 1) MediaPipe Pose 取 2D 關鍵點
# 2) 單一軸線：full/limb/fixed（原樣保存）
# 3) 兩段肢體線 + 肘/膝彎曲角度（原樣保存）
# 4) 新增：draw_compass_overlay() 在骨盆附近畫半環羅盤 + 指針 + 角度數字（H/V）

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import math
import numpy as np
import cv2
import os

# ---------------- FreeType 文字顯示（可選） ----------------
_FT = None
def _try_init_freetype():
    global _FT
    if _FT is not None:
        return _FT
    try:
        ft = cv2.freetype.createFreeType2()
        candidates = [
            Path(__file__).with_name("NotoSans-Regular.ttf"),
            Path(__file__).with_name("NotoSansSC-Regular.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ]
        font_path = None
        for p in candidates:
            if p.exists():
                font_path = p
                break
        if font_path is None:
            _FT = None
            return None
        ft.loadFontData(fontFileName=str(font_path), id=0)
        _FT = ft
        return _FT
    except Exception:
        _FT = None
        return None

def _put_text_any(img, text, org, font_scale=0.7, color=(0,0,0), thickness=2):
    """
    有 FreeType + 字型時能顯示 Unicode（°、中文）；否則回退 cv2.putText，
    並將 '°' 轉為 ASCII 的 ' deg' 避免顯示成 '??'。
    """
    ft = _try_init_freetype()
    if ft is not None:
        # FreeType 使用像素高度
        font_height = max(14, int(round(24 * font_scale)))
        try:
            ft.putText(img, text, org, font_height, color, thickness, cv2.LINE_AA, False)
            return
        except Exception:
            pass
    # safe_text = text.replace("°", " deg")
    # cv2.putText(img, safe_text, cv2.FONT_HERSHEY_SIMPLEX,
    #             font_scale, color, thickness, cv2.LINE_AA, org)\

    # ---- 這裡是關鍵修正：cv2.putText 的參數順序必須是 (img, text, org, fontFace, fontScale, color, ...) ----
    safe_text = text.replace("°", " deg")
    x, y = int(org[0]), int(org[1])  # 確保是整數座標
    cv2.putText(img, safe_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def _put_text_bg(img, text, org, font_scale=0.7,
                 text_color=(0,0,0), bg_color=(255,255,255),
                 thickness=2, padding=3, alpha=0.6):
    (tw, th), bl = cv2.getTextSize(text.replace("°"," deg"),
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + bl + padding
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    _put_text_any(img, text, (x, y), font_scale=font_scale, color=text_color, thickness=thickness)

# ---------------- Pose Estimator ----------------
class PoseEstimator:
    def __init__(self,
                 model_complexity: int = 0,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except Exception as e:
            raise ImportError("找不到 mediapipe，請先安裝（Python 3.12 建議改用 3.11）。") from e
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
        )
        self.LM = mp.solutions.pose.PoseLandmark

    def infer(self, frame_bgr) -> Dict[str, Tuple[int,int,float]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        out: Dict[str, Tuple[int,int,float]] = {}
        if not res.pose_landmarks:
            return out
        lm = res.pose_landmarks.landmark
        def put(name, idx):
            x = int(np.clip(lm[idx].x*w, 0, w-1))
            y = int(np.clip(lm[idx].y*h, 0, h-1))
            out[name] = (x, y, float(lm[idx].visibility))
        for name, idx in {
            "left_shoulder": self.LM.LEFT_SHOULDER,
            "left_elbow": self.LM.LEFT_ELBOW,
            "left_wrist": self.LM.LEFT_WRIST,
            "left_hip": self.LM.LEFT_HIP,
            "left_knee": self.LM.LEFT_KNEE,
            "left_ankle": self.LM.LEFT_ANKLE,
            "right_shoulder": self.LM.RIGHT_SHOULDER,
            "right_elbow": self.LM.RIGHT_ELBOW,
            "right_wrist": self.LM.RIGHT_WRIST,
            "right_hip": self.LM.RIGHT_HIP,
            "right_knee": self.LM.RIGHT_KNEE,
            "right_ankle": self.LM.RIGHT_ANKLE,
        }.items():
            put(name, idx)
        return out

# ---------------- 單一軸線工具 ----------------
@dataclass
class LineResult:
    p1: Tuple[int,int]
    p2: Tuple[int,int]
    ang_h_deg: float
    ang_v_deg: float
    anchor: Tuple[int,int]
    length_px: float

def _angles_from_direction(dx: float, dy: float) -> Tuple[float,float]:
    # 影像座標 y 向下 => atan2(-dy, dx)；回傳與水平/垂直的夾角（0~90）
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return (float("nan"), float("nan"))
    theta = abs(math.degrees(math.atan2(-dy, dx))) % 180.0  # 0~180
    if theta > 90.0: theta = 180.0 - theta
    return (theta, abs(90.0-theta))

def _extend_line_through_point(anchor, direction, w, h):
    ax, ay = anchor; dx, dy = direction
    if abs(dx) < 1e-6 and abs(dy) < 1e-6: return None
    t = []
    if abs(dx) > 1e-6: t += [(0-ax)/dx, ((w-1)-ax)/dx]
    if abs(dy) > 1e-6: t += [(0-ay)/dy, ((h-1)-ay)/dy]
    pts = []
    for ti in t:
        x = ax + ti*dx; y = ay + ti*dy
        if -1 <= x <= w and -1 <= y <= h:
            pts.append((int(np.clip(round(x),0,w-1)), int(np.clip(round(y),0,h-1))))
    if len(pts) < 2:
        if abs(dx) < 1e-6: pts = [(ax,0),(ax,h-1)]
        elif abs(dy) < 1e-6: pts = [(0,ay),(w-1,ay)]
        else: return None
    def d2(a,b): return (a[0]-b[0])**2+(a[1]-b[1])**2
    p1, p2, best = pts[0], pts[1], d2(pts[0],pts[1])
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            dd = d2(pts[i],pts[j])
            if dd > best: p1,p2,best = pts[i],pts[j],dd
    return p1, p2

def _segment_around_anchor(anchor, direction, length_px, w, h):
    ax, ay = anchor; dx, dy = direction
    n = math.hypot(dx, dy)
    if n < 1e-6: return None
    ux, uy = dx/n, dy/n
    half = length_px/2.0
    p1 = (int(np.clip(round(ax-ux*half),0,w-1)), int(np.clip(round(ay-uy*half),0,h-1)))
    p2 = (int(np.clip(round(ax+ux*half),0,w-1)), int(np.clip(round(ay+uy*half),0,h-1)))
    return p1, p2

def make_axis_line(frame_shape, upstream, anchor, downstream, mode="full", fixed_len_px=None) -> Optional[LineResult]:
    h, w = frame_shape[:2]
    dx, dy = downstream[0]-upstream[0], downstream[1]-upstream[1]
    ang_h, ang_v = _angles_from_direction(dx, dy)
    if mode == "full":
        ends = _extend_line_through_point(anchor, (dx,dy), w, h)
        if ends is None: return None
        L = math.hypot(ends[0][0]-ends[1][0], ends[0][1]-ends[1][1])
        return LineResult(ends[0], ends[1], ang_h, ang_v, anchor, L)
    if mode == "limb":
        L = math.hypot(dx, dy)
        if L < 1e-6: return None
        ends = _segment_around_anchor(anchor, (dx,dy), L, w, h)
        return LineResult(ends[0], ends[1], ang_h, ang_v, anchor, L)
    if mode == "fixed":
        L = float(fixed_len_px if fixed_len_px and fixed_len_px > 1 else 200.0)
        ends = _segment_around_anchor(anchor, (dx,dy), L, w, h)
        return LineResult(ends[0], ends[1], ang_h, ang_v, anchor, L)
    return None

# ---------------- 兩段肢體線 + 肘/膝角度 ----------------
@dataclass
class JointAngleResult:
    joint: str
    a: Tuple[int,int]
    joint_xy: Tuple[int,int]
    c: Tuple[int,int]
    angle_deg: float
    len_proximal: float
    len_distal: float

def _angle_three_points(a,b,c) -> float:
    ux, uy = a[0]-b[0], a[1]-b[1]
    vx, vy = c[0]-b[0], c[1]-b[1]
    nu, nv = math.hypot(ux,uy), math.hypot(vx,vy)
    if nu < 1e-6 or nv < 1e-6: return float("nan")
    cosv = (ux*vx + uy*vy)/(nu*nv)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def _theta_from_vec(dx, dy) -> float:
    return (math.degrees(math.atan2(-dy, dx)) % 360.0)

def _shortest_delta(a,b) -> float:
    return (b - a + 180.0) % 360.0 - 180.0

def _draw_angle_arc(img, center, a_pt, c_pt, radius=35, color=(255,255,255), thickness=2):
    cx, cy = center
    th1 = _theta_from_vec(a_pt[0]-cx, a_pt[1]-cy)
    th2 = _theta_from_vec(c_pt[0]-cx, c_pt[1]-cy)
    d = _shortest_delta(th1, th2)
    N = max(8, int(abs(d)//5)+1)
    pts = []
    for i in range(N+1):
        t = i/N
        th = th1 + d*t
        rad = math.radians(th)
        x = int(round(cx + radius*math.cos(rad)))
        y = int(round(cy - radius*math.sin(rad)))
        pts.append((x,y))
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)

def draw_joint_pairs_and_angles(frame_bgr, kps, vis_thresh=0.5, draw_angles=True) -> Dict[str, Optional[JointAngleResult]]:
    out = {"left_elbow": None, "right_elbow": None, "left_knee": None, "right_knee": None}
    def ok(n): return (n in kps) and (kps[n][2] >= vis_thresh)
    specs = [
        ("left_elbow",  "left_shoulder","left_elbow","left_wrist",(0,255,255),(0,200,255)),
        ("right_elbow", "right_shoulder","right_elbow","right_wrist",(0,255,0),(0,200,0)),
        ("left_knee",   "left_hip","left_knee","left_ankle",(255,255,0),(200,200,0)),
        ("right_knee",  "right_hip","right_knee","right_ankle",(255,200,0),(255,160,0)),
    ]
    H, W = frame_bgr.shape[:2]
    base_radius = int(max(20, min(H,W)*0.035))
    for key, prox, joint, dist, c1, c2 in specs:
        if ok(prox) and ok(joint) and ok(dist):
            A, B, C = kps[prox][:2], kps[joint][:2], kps[dist][:2]
            ang = _angle_three_points(A,B,C)
            cv2.line(frame_bgr, A, B, c1, 3, cv2.LINE_AA)
            cv2.line(frame_bgr, B, C, c2, 3, cv2.LINE_AA)
            cv2.circle(frame_bgr, B, 5, (255,255,255), -1, cv2.LINE_AA)
            if draw_angles and math.isfinite(ang):
                _draw_angle_arc(frame_bgr, B, A, C, radius=base_radius, color=(255,255,255), thickness=2)
                th1 = _theta_from_vec(A[0]-B[0], A[1]-B[1])
                th2 = _theta_from_vec(C[0]-B[0], C[1]-B[1])
                mid = th1 + _shortest_delta(th1, th2)/2.0
                rad = math.radians(mid)
                tx = int(round(B[0] + (base_radius+14)*math.cos(rad)))
                ty = int(round(B[1] - (base_radius+14)*math.sin(rad)))
                _put_text_bg(frame_bgr, f"{ang:.1f}°", (tx, ty), font_scale=0.7)
            out[key] = JointAngleResult(key, A, B, C, ang,
                                        math.hypot(A[0]-B[0],A[1]-B[1]),
                                        math.hypot(C[0]-B[0],C[1]-B[1]))
    return out

# ---------------- 羅盤（半環刻度 + 指針 + H/V 數字） ----------------
def _ring_band(img, center, r_in, r_out, start_deg, end_deg, color, alpha=0.35, steps=120):
    cx, cy = center
    start_rad = math.radians(start_deg); end_rad = math.radians(end_deg)
    N = max(8, int(abs(end_deg-start_deg)//2)+1, steps)
    inner, outer = [], []
    for i in range(N+1):
        t = i/N; th = start_rad + (end_rad-start_rad)*t
        outer.append((int(round(cx + r_out*math.cos(th))),
                      int(round(cy - r_out*math.sin(th)))))
    for i in range(N, -1, -1):
        t = i/N; th = start_rad + (end_rad-start_rad)*t
        inner.append((int(round(cx + r_in*math.cos(th))),
                      int(round(cy - r_in*math.sin(th)))))
    poly = np.array(outer+inner, np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def _tick_ring(img, center, r1, r2, start_deg, end_deg, major=30, minor=6,
               color_major=(240,240,240), color_minor=(180,180,180), th_major=2, th_minor=1):
    cx, cy = center
    def draw_one(step, col, th):
        deg = start_deg
        while deg <= end_deg + 1e-6:
            rad = math.radians(deg)
            x1, y1 = int(round(cx + r1*math.cos(rad))), int(round(cy - r1*math.sin(rad)))
            x2, y2 = int(round(cx + r2*math.cos(rad))), int(round(cy - r2*math.sin(rad)))
            cv2.line(img, (x1,y1), (x2,y2), col, th, cv2.LINE_AA)
            deg += step
    draw_one(minor, color_minor, th_minor)
    draw_one(major, color_major, th_major)

def draw_compass_overlay(
    frame_bgr,
    kps: Dict[str, Tuple[int,int,float]],
    prefer: str = "right_leg",
    vis_thresh: float = 0.5,
    *,
    visible: bool = True,        # 整個羅盤是否顯示
    show_needle: bool = True,    # 是否顯示紅色指針
    hv_mode: str = "both"        # "both" | "h" | "v" | "none"
):
    """
    羅盤式角度顯示（不改你的線）：
    - 中心：左右髖中點（缺一側則用另一側；都缺用畫面中心）
    - 向量：預設右腿 (right_hip->right_ankle)，再 fallback 到左腿 / 右臂 / 左臂
    - 顯示：半環(210°~390°) + 刻度 + (可選)指針 + (可選) H/V 數字
    - 角度計算：y 向下 → atan2(-dy, dx)，H∈[0,90]，V=90-H（避免除以零做 1e-6 防護）
    """
    if not visible:
        return

    H, W = frame_bgr.shape[:2]
    def ok(n): return (n in kps) and (kps[n][2] >= vis_thresh)
    def pt(n): return kps[n][:2]

    # 羅盤中心
    if ok("left_hip") and ok("right_hip"):
        cx = int(round((pt("left_hip")[0]+pt("right_hip")[0])*0.5))
        cy = int(round((pt("left_hip")[1]+pt("right_hip")[1])*0.5))
    elif ok("left_hip"):
        cx, cy = pt("left_hip")
    elif ok("right_hip"):
        cx, cy = pt("right_hip")
    else:
        cx, cy = W//2, H//2

    # 方向來源優先序
    order = [("right_hip","right_ankle"), ("left_hip","left_ankle"),
             ("right_shoulder","right_wrist"), ("left_shoulder","left_wrist")] \
            if prefer == "right_leg" else \
            [("left_hip","left_ankle"), ("right_hip","right_ankle"),
             ("left_shoulder","left_wrist"), ("right_shoulder","right_wrist")]
    dir_vec = None
    for a,b in order:
        if ok(a) and ok(b):
            ax, ay = pt(a); bx, by = pt(b)
            dir_vec = (bx-ax, by-ay)
            break
    if dir_vec is None:
        return

    dx, dy = dir_vec
    h_deg, v_deg = _angles_from_direction(dx, dy)

    # 幾何
    R = int(min(H, W) * 0.42)
    rin, rout = int(R*0.78), int(R*0.98)
    start_deg, end_deg = 210, 390

    # 半環 + 刻度
    _ring_band(frame_bgr, (cx, cy), rin, rout, start_deg, end_deg, (180,180,180), alpha=0.30)
    _ring_band(frame_bgr, (cx, cy), rin+2, rout-2, start_deg+10, end_deg-10, (0,0,255), alpha=0.22)
    _tick_ring(frame_bgr, (cx, cy), rin-8, rin-2, start_deg, end_deg, major=30, minor=6)

    # 指針
    if show_needle:
        theta = _theta_from_vec(dx, dy)
        rad = math.radians(theta)
        px = int(round(cx + (rout-4) * math.cos(rad)))
        py = int(round(cy - (rout-4) * math.sin(rad)))
        cv2.line(frame_bgr, (cx, cy), (px, py), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

    # 角度字串（依 hv_mode）
    mode = (hv_mode or "both").lower()
    txt = None
    if mode == "both":
        txt = f"H {h_deg:.1f}° / V {v_deg:.1f}°"
    elif mode == "h":
        txt = f"H {h_deg:.1f}°"
    elif mode == "v":
        txt = f"V {v_deg:.1f}°"
    elif mode == "none":
        txt = None

    if txt:
        _put_text_bg(frame_bgr, txt,
                     (cx + int(R*0.05), cy - int(R*0.1)),
                     font_scale=0.75, text_color=(255,255,255),
                     bg_color=(0,0,0), alpha=0.55)

# （保留）畫軸線（供你原先流程使用）
def draw_axes_on_frame(frame_bgr, kps, vis_thresh=0.5, line_mode="full", fixed_len_px=None):
    out = {"left_arm": None, "right_arm": None, "left_leg": None, "right_leg": None}
    def ok(n): return (n in kps) and (kps[n][2] >= vis_thresh)
    pairs = [
        ("left_arm","left_shoulder","left_elbow","left_wrist",(0,255,255)),
        ("right_arm","right_shoulder","right_elbow","right_wrist",(0,255,0)),
        ("left_leg","left_hip","left_knee","left_ankle",(255,255,0)),
        ("right_leg","right_hip","right_knee","right_ankle",(255,200,0)),
    ]
    for key, up, mid, dn, color in pairs:
        if ok(up) and ok(mid) and ok(dn):
            lr = make_axis_line(frame_bgr.shape, kps[up][:2], kps[mid][:2], kps[dn][:2],
                                mode=line_mode, fixed_len_px=fixed_len_px)
            if lr is not None:
                cv2.line(frame_bgr, lr.p1, lr.p2, color, 3, cv2.LINE_AA)
                cv2.circle(frame_bgr, lr.anchor, 4, color, -1, cv2.LINE_AA)
                _put_text_bg(frame_bgr, f"H {lr.ang_h_deg:.1f}° / V {lr.ang_v_deg:.1f}°",
                             (lr.anchor[0]+8, lr.anchor[1]-8))
            out[key] = lr
    return out
