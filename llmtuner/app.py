import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from llmtuner.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
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
    return app.exec()


def cli_main():
    sys.exit(main())
