import sys
from importlib import resources

import pyqtgraph as pg
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from qt_material import build_stylesheet

import miniml.resources
from miniml.gui.windows import AppMainWindow

# ------- GUI config ------- #
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("leftButtonPan", False)


def entry_point():
    app = QApplication(sys.argv)

    with resources.as_file(
        resources.files(miniml.resources) / "minML_icon.png"
    ) as app_icon_file:
        app.setWindowIcon(QIcon(str(app_icon_file)))

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

    window = AppMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    entry_point()
