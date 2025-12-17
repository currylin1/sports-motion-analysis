# Python 3.12 / 最簡單可播放影片的程式（OpenCV）
import cv2
from pathlib import Path
import sys
import math

def main():
    base = Path(__file__).resolve().parent
    # 預設讀取 專案/測試影片/demo_input.mp4
    video_path = base / "測試影片" / "123.mp4"
    # 也支援從命令列帶入路徑：python play_video.py <你的影片路徑>
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])

    if not video_path.exists():
        raise FileNotFoundError(f"找不到影片：{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片（可能是路徑或編碼器問題）：{video_path}")

    # 嘗試以檔案內建 FPS 播放；若取得失敗則採用 33ms (~30fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = 33
    if fps and math.isfinite(fps) and fps > 0:
        delay_ms = max(int(1000 / fps), 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
    print(f"[INFO] 路徑: {video_path}")
    print(f"[INFO] 解析度: {width}x{height}, FPS: {fps:.2f}" if fps else "[INFO] FPS: 未知")
    print(f"[INFO] 總幀數: {total if total > 0 else '未知'}")

    # 可調整大小的視窗；按 Space 暫停/繼續，Esc 或 q 離開
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # 讀不到新幀：可能播完或讀檔異常
                break
            cv2.imshow("Video", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord('q')):  # Esc 或 'q'
                break
            if key == 32:  # Space 暫停
                # 等待下一個按鍵：Space 繼續，Esc/q 離開
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 in (32, 27, ord('q')):
                        break
                if key2 in (27, ord('q')):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
