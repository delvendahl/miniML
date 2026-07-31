import sys
from importlib import resources

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from qt_material import build_stylesheet

import miniml.resources
from miniml.gui.presenters import MainWindowPresenter
from miniml.gui.services import AppServices
from miniml.gui.state import AppState
from miniml.gui.views import MainWindow
from miniml.resources.util import get_app_icon_file_path

# ------- GUI config ------- #
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("leftButtonPan", False)


def entry_point():
    """
    Initialize and launch the miniML GUI application.
    """
    # On Windows, set the application ID to correctly show app icon in the taskbar
    if sys.platform == "win32":
        try:
            import ctypes

            app_id = "org.miniML.app"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to set application ID: {e}", file=sys.stderr)

    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app.setWindowIcon(QIcon(get_app_icon_file_path()))

    extra = {
        "density_scale": "-1",
    }

    with resources.as_file(
        resources.files(miniml.resources) / "miniml.css.template"
    ) as template_file:
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")

        app.setStyleSheet(
            build_stylesheet(
                theme="light_blue.xml",
                invert_secondary=False,
                extra=extra,
                template=str(template_file),
            )
        )

    state = AppState()
    services = AppServices()
    window = MainWindow(state=state, services=services)
    presenter = MainWindowPresenter(state=state, services=services, parent=window)
    presenter.bind_view(window)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    entry_point()
