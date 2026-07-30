import pyqtgraph as pg
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout


def finalize_dialog_window(
    window: QDialog, title: str = "new window", cancel: bool = True
) -> None:
    """Finalize a dialog with standard buttons and modality."""
    qbtn = (
        (QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        if cancel
        else QDialogButtonBox.Close
    )
    window.buttonBox = QDialogButtonBox(qbtn)
    if cancel:
        window.buttonBox.accepted.connect(window.accept)
        window.buttonBox.rejected.connect(window.reject)
    else:
        window.buttonBox.clicked.connect(window.accept)

    layout = window.layout()
    if isinstance(layout, QFormLayout):
        layout.addRow(window.buttonBox)
    window.setWindowTitle(title)
    window.setWindowModality(pg.QtCore.Qt.WindowModality.ApplicationModal)
