import pyabf
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDialog, QFormLayout, QLineEdit

from miniml.fileio import heka_reader as heka
from miniml.fileio.util import get_hdf_keys
from miniml.gui.dialogs.common import finalize_dialog_window


class LoadAbfPanel(QDialog):
    """
    Dialog that collects AXON ABF loading parameters.
    """

    def __init__(self, filename: str, parent=None):
        """
        Initialize ABF controls from file metadata.
        """
        super().__init__(parent)

        self.abf_file = pyabf.ABF(filename)

        self.channel = QComboBox()
        self.channel.addItems([str(channel) for channel in self.abf_file.channelList])
        self.channel.setMinimumWidth(150)
        self.channel.currentIndexChanged[str].connect(
            self.on_comboBoxParent_currentChannelChanged
        )

        self.scale = QLineEdit("1")
        self.unit = QLineEdit(self.abf_file.adcUnits[0])
        self.protocol = QLineEdit(self.abf_file.protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setMinimumWidth(300)

        layout = QFormLayout(self)
        layout.addRow("Recording channel:", self.channel)
        layout.addRow("Scaling factor:", self.scale)
        layout.addRow("Data unit:", self.unit)
        layout.addRow("Protocol:", self.protocol)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load AXON .abf file")

    @pyqtSlot(str)
    def on_comboBoxParent_currentChannelChanged(self, index):
        """
        Update the unit field when the selected ABF channel changes.
        """
        self.unit.clear()
        self.unit.setText(self.abf_file.adcUnits[int(index)])


class LoadHdfPanel(QDialog):
    """
    Dialog that collects HDF5 dataset loading parameters.
    """

    def __init__(self, filename: str, parent=None):
        """
        Initialize HDF5 controls and available dataset keys.
        """
        super().__init__(parent)

        self.e1 = QComboBox()
        self.e1.setMinimumWidth(200)
        self.e1.addItems(get_hdf_keys(filename))
        self.e2 = QLineEdit("2e-5")
        self.e2.setMinimumWidth(200)
        self.e3 = QLineEdit("1e12")
        self.e3.setMinimumWidth(200)
        self.e4 = QLineEdit("pA")
        self.e4.setMinimumWidth(200)

        layout = QFormLayout(self)
        layout.addRow("Dataset name:", self.e1)
        layout.addRow("Sampling interval (s):", self.e2)
        layout.addRow("Scaling factor:", self.e3)
        layout.addRow("Data unit:", self.e4)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load HDF .h5 file")


class LoadDatPanel(QDialog):
    """
    Dialog that collects HEKA DAT loading parameters.
    """

    def __init__(self, filename: str, parent=None):
        """
        Initialize DAT controls from available groups and series.
        """
        super().__init__(parent)

        self.bundle = heka.Bundle(filename)

        group_series = []
        for i, GroupRecord in enumerate(self.bundle.pul.children):
            group_series.append(str(i + 1) + " - " + GroupRecord.Label)
        self.group = QComboBox()
        self.group.addItems(group_series)
        self.group.setMinimumWidth(150)
        self.group.currentIndexChanged[str].connect(
            self.on_comboBoxParent_currentIndexChanged
        )

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[0].children):
            bundle_series.append(str(i + 1) + " - " + SeriesRecord.Label)
        self.series = QComboBox()
        self.series.addItems(bundle_series)
        self.series.setMinimumWidth(300)
        self.load_option = QCheckBox("")
        self.e1 = QLineEdit("")
        self.e2 = QLineEdit("1e12")
        self.e3 = QLineEdit("pA")

        layout = QFormLayout(self)
        layout.addRow("Import group:", self.group)
        layout.addRow("Import series:", self.series)
        layout.addRow("Import all series of this type:", self.load_option)
        layout.addRow("Exclude selected series:", self.e1)
        layout.addRow("Scaling factor:", self.e2)
        layout.addRow("Data unit:", self.e3)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load HEKA .dat file")
        self.finished.connect(self.on_dialog_finished)

    @pyqtSlot(str)
    def on_comboBoxParent_currentIndexChanged(self, index):
        """
        Refresh the series list when the selected group changes.
        """
        group_no, _ = index.split(" - ")

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[int(group_no) - 1].children):
            bundle_series.append(str(i + 1) + " - " + SeriesRecord.Label)

        self.series.clear()
        self.series.addItems(bundle_series)

    @pyqtSlot()
    def on_dialog_finished(self):
        """
        Release the opened HEKA bundle when the dialog closes.
        """
        self.bundle.close()
