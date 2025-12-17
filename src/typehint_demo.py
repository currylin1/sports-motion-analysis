# typehint_demo.py (Python 3.12)
from pathlib import Path
from typing import get_type_hints
import sys

def RUNTIME_BASE() -> Path:
    # 只是示意，實務上你可能用 sys._MEIPASS 等
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

def bad() -> int:
    # 故意回傳錯誤型別（字串而不是 int），Python 執行期仍會跑過
    return "not an int"  # type: ignore[return-value]

if __name__ == "__main__":
    print("annotations dict =", RUNTIME_BASE.__annotations__)
    print("get_type_hints   =", get_type_hints(RUNTIME_BASE))
    print("RUNTIME_BASE()   =", RUNTIME_BASE())

    # Python 不會因為型別不符而丟錯（除非你自己檢查）
    val = bad()
    print("bad() returned   =", val, "(type:", type(val), ")")

    # 若想在執行期自檢，可手動 assert：
    assert isinstance(RUNTIME_BASE(), Path), "RUNTIME_BASE 沒回傳 Path！"
