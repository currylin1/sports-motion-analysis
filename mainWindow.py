# mainWindow.py — Python 3.12 / PySide6
from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSettings, QProcess, QTimer, Slot,QDir
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QVBoxLayout, QLabel, QSizePolicy, QWidget,QFileSystemModel
)

from ui.ui_mainwindow import Ui_MainWindow  # 依你的專案調整


# -------------------- 小工具：OpenCV BGR -> QPixmap --------------------
def cvimg_to_qpixmap(frame_bgr: Optional[np.ndarray]) -> QPixmap:
    if frame_bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -------------------- QProcess：dev/frozen 自動切換 --------------------
def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)

def program_and_args_for_worker(video_path: str):
    """
    回傳 (program, args)
      - 開發時：program=python.exe, args=[play_with_axes.py, ...]
      - 打包後：program=play_with_axes.exe, args=[--video, ...]
    """
    base = Path(sys.executable).parent if _is_frozen() else Path(__file__).resolve().parent
    exe = base / "play_with_axes.exe"
    if exe.exists():  # 打包後
        return str(exe), ["--video", video_path, "--no-display"]
    else:             # 開發模式
        py = base / "play_with_axes.py"
        return sys.executable, [str(py), "--video", video_path, "--no-display", "--remove-bg"]


# -------------------- 可選：用 QWidget 自行等比繪製（目前先用 QLabel） --------------------
class VideoCanvas(QWidget):
    """若日後要改成自繪影音面板可用此類別；目前 GUI 用 QLabel 即可"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = QPixmap()

    def set_pixmap(self, pix: QPixmap):
        self._pix = pix if not pix.isNull() else QPixmap()
        self.update()

    def set_ndarray(self, frame_bgr):
        self.set_pixmap(cvimg_to_qpixmap(frame_bgr))

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if not self._pix.isNull():
            target = self.contentsRect().size()
            scaled = self._pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        p.end()


# ======================================================================
# MainWindow
# ======================================================================
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # 狀態
        self.selected_video: Optional[Path] = None
        self.proc: Optional[QProcess] = None
        self._proc_buffer = ""                  # 收集 stdout
        self._last_out_video: Optional[Path] = None

        # ---- 檔案路徑顯示 ----
        self.lblPath.setText("尚未選擇影片")
        self.lblPath.setWordWrap(False)
        self.lblPath.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # --- 讓 tvFiles 顯示家目錄 ---
        self._fsModel = QFileSystemModel(self)
        self._fsModel.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)

        # 先顯示全部檔案，確定有畫面（之後再加副檔名篩選）
        self._fsModel.setNameFilters(["*.mp4", "*.mov", "*.m4v", "*.avi", "*.mkv"])
        self._fsModel.setNameFilterDisables(False)  # 不符合濾鏡的檔案會被隱藏
        # 👉 預設就指到 專案/待處理影片（不存在則退回家目錄）
        default_root = self._inbox_dir()
        root_str = str(default_root if default_root.exists() else Path.home())

        root_dir = str(Path.home())
        idx = self._fsModel.setRootPath(root_str)

        self.tvFiles.setModel(self._fsModel)
        self.tvFiles.setRootIndex(idx)
        self.tvFiles.setAnimated(True)
        self.tvFiles.setSortingEnabled(True)
        self.tvFiles.sortByColumn(0, Qt.AscendingOrder)

        # 只留「名稱」欄
        for col in (1, 2, 3):
            self.tvFiles.setColumnHidden(col, True)

        # 👉 訊號（若尚未連線）
        self.tvFiles.selectionModel().currentChanged.connect(self._on_tree_selected)
        self.tvFiles.doubleClicked.connect(self._on_tree_double_clicked)

        # ---- 右側影片預覽面板（用 QLabel 顯示影格）----
        self.videoPanel = self.lblVido  # 你的 UI 內的容器 widget
        self.videoLabel = QLabel("尚未播放", self.videoPanel)
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("background:#000; color:#aaa; border:1px solid #333;")
        lay = QVBoxLayout(self.videoPanel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.videoLabel)

        self.videoLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.videoLabel.setMinimumSize(1, 1)
        self.videoLabel.setScaledContents(False)  # 我們用程式做等比縮放

        # 播放器資源
        self._cap: Optional[cv2.VideoCapture] = None
        self._timer: Optional[QTimer] = None
        self._last_frame: Optional[np.ndarray] = None
        self._is_paused = False

        # 事件連結
        self.btnBrowse.clicked.connect(self.choose_root_dir)
        if hasattr(self, "btnStartAlysis"):
            self.btnStartAlysis.clicked.connect(self.start_analysis)
        if hasattr(self, "btnStop"):
            self.btnStop.clicked.connect(self.toggle_play_pause)
            self.btnStop.setText("暫停播放")
            self.btnStop.setEnabled(False)

    def _update_path_label(self, full_text: str):
        fm = self.lblPath.fontMetrics()
        maxw = max(40, self.lblPath.width() - 8)
        elided = fm.elidedText(full_text, Qt.ElideMiddle, maxw)
        self.lblPath.setText(elided)
        self.lblPath.setToolTip(full_text)

    def _on_tree_selected(self, current, _prev):
        """單擊：若選到影片檔，更新路徑標籤與內部選取變數。"""
        path = Path(self._fsModel.filePath(current))
        if path.is_file():
            self.selected_video = path
            self._update_path_label(str(path))

    def _on_tree_double_clicked(self, index):
        """雙擊：若是影片檔，直接開始分析。"""
        path = Path(self._fsModel.filePath(index))
        if path.is_file():
            self.selected_video = path
            self._update_path_label(str(path))
            if hasattr(self, "start_analysis"):
                self.start_analysis()
    # === 新增：類別內私有工具方法 ===
    def _app_base_dir(self) -> Path:
        """開發：回到此檔所在資料夾；打包：回到 .exe 同層資料夾。"""
        return Path(sys.executable).parent if getattr(sys, "frozen", False) \
            else Path(__file__).resolve().parent

    def _inbox_dir(self) -> Path:
        """專案中的『待處理影片』資料夾；若不存在則建立。"""
        p = self._app_base_dir() / "待處理影片"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _current_root_dir(self) -> str:
        """取得目前 TreeView 的根目錄，沒有就回家目錄。"""
        idx = self.tvFiles.rootIndex()
        try:
            # _fsModel 是你前一步建立的 QFileSystemModel
            p = self._fsModel.filePath(idx)
            return p if p else str(Path.home())
        except Exception:
            return str(Path.home())

    @Slot()
    def choose_root_dir(self):
        """開對話框，選新根目錄並刷新 tvFiles。"""
        start_dir = str(self._inbox_dir() if self._inbox_dir().exists() else self._current_root_dir())
        d = QFileDialog.getExistingDirectory(self, "選擇根目錄", start_dir)
        if not d:
            return
        idx = self.fsModel.setRootPath(d)
        self.tvFiles.setRootIndex(idx)
        if hasattr(self, "lblPath"):
            self.lblPath.setText(d)
            self.lblPath.setToolTip(d)
    # -------------------- 視窗縮放：更新路徑省略 --------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.selected_video:
            self._update_path_label(str(self.selected_video))

    # -------------------- 選檔 --------------------
    def choose_file(self):
        start_dir = self._load_last_dir()
        filters = "影片檔 (*.mp4 *.mov *.m4v *.avi *.mkv);;所有檔案 (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", str(start_dir), filters)
        if not path:
            return
        p = Path(path)
        self.selected_video = p
        self._save_last_dir(p.parent)
        self._update_path_label(str(p))
        print(f"[選擇] {p} | 存在: {p.exists()}", flush=True)

    def _update_path_label(self, full_text: str):
        fm = self.lblPath.fontMetrics()
        max_width = max(40, self.lblPath.width() - 8)
        elided = fm.elidedText(full_text, Qt.ElideMiddle, max_width)
        self.lblPath.setText(elided)
        self.lblPath.setToolTip(full_text)

    # -------------------- QSettings --------------------
    def _settings(self) -> QSettings:
        return QSettings("winfly", "sports-vision-tool")

    def _load_last_dir(self) -> Path:
        s = self._settings()
        return Path(s.value("last_dir", str(Path.home())))

    def _save_last_dir(self, directory: Path):
        s = self._settings()
        s.setValue("last_dir", str(directory))

    def get_selected_video_path(self) -> Optional[str]:
        if self.selected_video:
            return str(self.selected_video)
        tip = self.lblPath.toolTip() or ""
        return tip if tip and tip != "尚未選擇影片" else None

    # -------------------- 子行程：呼叫 play_with_axes --------------------
    @Slot()
    def start_analysis(self):
        video_path = self.get_selected_video_path()
        self.stop_panel_player()
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(self, "提醒", "請先選擇一個有效的影片檔。")
            return

        # 關閉舊行程
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

        self._proc_buffer = ""
        self._last_out_video = None
        self.stop_panel_player()  # 避免占用輸出檔

        # 取得對應的 program/args（dev: python + .py；打包: play_with_axes.exe）
        program, args = program_and_args_for_worker(video_path)

        self.proc = QProcess(self)
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(str(Path(program).parent))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)

        if hasattr(self, "btnStartAlysis"):
            self.btnStartAlysis.setEnabled(False)
            self.btnStartAlysis.setText("處理中…")

        print(f"[執行] {program} {' '.join(args)}", flush=True)
        self.proc.start()

    def _on_proc_output(self):
        if not self.proc:
            return
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        self._proc_buffer += text
        print(text, end="", flush=True)

        # 解析輸出影片路徑（支援兩種訊息行）
        for pat in (r"\[OK\]\s*已保留原始聲音：(.+)", r"\[OK\]\s*影片輸出：(.+)"):
            m = re.search(pat, text)
            if m:
                outp = Path(m.group(1).strip().strip('"')).expanduser()
                self._last_out_video = outp
                break

    def _on_proc_finished(self, exitCode: int, exitStatus):
        if hasattr(self, "btnStartAlysis"):
            self.btnStartAlysis.setEnabled(True)
            self.btnStartAlysis.setText("開始分析")

        if exitCode != 0:
            QMessageBox.critical(self, "失敗", f"分析腳本結束碼：{exitCode}\n請查看主控台訊息。")
            self.proc = None
            return

        # 優先用 stdout 解析到的路徑；若沒有，嘗試預測檔名
        out_path: Optional[Path] = self._last_out_video
        if not out_path:
            in_path = Path(self.get_selected_video_path() or "")
            cand = Path(program_and_args_for_worker(in_path.as_posix())[0]).parent / "輸出影片" / f"{in_path.stem}_with_compass.mp4"
            if cand.exists():
                out_path = cand

        if out_path and out_path.exists():
            QMessageBox.information(self, "完成", f"分析完成！\n將在右側面板播放：\n{out_path}")
            self.open_video_for_panel(str(out_path))
        else:
            QMessageBox.information(self, "完成", "分析完成，但找不到輸出影片路徑。\n請查看主控台輸出。")

        self.proc = None

    # -------------------- 右側面板播放器（循環播放） --------------------
    def open_video_for_panel(self, video_path: str):
        self.stop_panel_player()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(self, "播放失敗", f"無法開啟影片：\n{video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(int(1000 / fps), 1)

        self._cap = cap
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer.start(interval)
        self._last_frame = None
        self._is_paused = False
        self.videoLabel.setText("")

        # 啟用暫停鍵
        if hasattr(self, "btnStop"):
            self.btnStop.setEnabled(True)
            self.btnStop.setText("暫停播放")

    def _on_timer_tick(self):
        if not self._cap:
            return
        ok, frame = self._cap.read()
        if not ok:
            # 播到尾端就循環
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        self._last_frame = frame
        pix = cvimg_to_qpixmap(frame)
        scaled = pix.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.videoLabel.setPixmap(scaled)

    def stop_panel_player(self):
        if self._timer:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._last_frame = None
        self.videoLabel.setPixmap(QPixmap())
        self.videoLabel.setText("尚未播放")
        self._is_paused = False

        # 停用暫停鍵
        if hasattr(self, "btnStop"):
            self.btnStop.setEnabled(False)
            self.btnStop.setText("暫停播放")

    @Slot()
    def toggle_play_pause(self):
        if not self._cap or not self._timer:
            QMessageBox.information(self, "提示", "目前沒有正在播放的影片。")
            return

        if not self._is_paused:
            self._timer.stop()
            self._is_paused = True
            if hasattr(self, "btnStop"):
                self.btnStop.setText("繼續播放")
        else:
            interval = self._timer.interval()
            if not interval:
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                interval = max(int(1000 / fps), 1)
            self._timer.start(interval)
            self._is_paused = False
            if hasattr(self, "btnStop"):
                self.btnStop.setText("暫停播放")


# -------------------- 入口 --------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle("運動分析程式")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
