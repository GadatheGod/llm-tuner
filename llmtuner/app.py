import sys
import os
import ctypes
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from llmtuner.ui.main_window import MainWindow

kernel32 = ctypes.windll.kernel32
MUTEX_NAME = "Global\\LLM-Tuner_Mutex"
MUTEX_HANDLE = None


def acquire_single_instance():
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, True, MUTEX_NAME)
    if not handle:
        return False
    last_error = kernel32.GetLastError()
    if last_error == 183:
        kernel32.CloseHandle(handle)
        return False
    global MUTEX_HANDLE
    MUTEX_HANDLE = handle
    return True


def release_single_instance():
    global MUTEX_HANDLE
    if MUTEX_HANDLE:
        kernel32.ReleaseMutex(MUTEX_HANDLE)
        kernel32.CloseHandle(MUTEX_HANDLE)
        MUTEX_HANDLE = None


def main():
    if os.name == "nt":
        if not acquire_single_instance():
            os._exit(1)

    existing = QApplication.instance()
    if existing and hasattr(existing, "llm_tuner_initialized"):
        acquire_single_instance()
        sys.exit(0)

    app = QApplication(sys.argv)
    app.llm_tuner_initialized = True
    app.setApplicationName("LLM-Tuner")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet("""
        QMenuBar::item:selected { background: #2d5f8a; color: white; }
        QMenuBar::item:pressed { background: #1a3d5c; }
        QPushButton {
            padding: 6px 16px;
            border-radius: 4px;
            border: 1px solid #2d5f8a;
            background: #2d5f8a;
            color: white;
        }
        QPushButton:hover { background: #3a7cb8; }
        QPushButton:pressed { background: #1a3d5c; }
        QTabBar::tab { padding: 8px 16px; }
        QTabBar::tab:selected { background: #2d5f8a; color: white; }
        QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    """)
    window = MainWindow()
    window.show()
    result = app.exec()

    if os.name == "nt":
        release_single_instance()
    return result


def cli_main():
    sys.exit(main())
