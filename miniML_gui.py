from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np
import os
import sys
sys.path.append('./core/')
from miniML import MiniTrace, EventDetection
from miniML_settings import MinimlSettings
from qt_material import build_stylesheet
import FileImport.HekaReader as heka
import tensorflow as tf


# ------- Functions ------- #
def get_available_models():
    """
    Returns a list of available model paths in the /models folder.
    The list only contains relative paths.
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    models = [os.path.relpath(os.path.join(root, file), models_dir)
              for root, dirs, files in os.walk(models_dir)
              for file in files
              if file.endswith('.h5')]
    
    return models


def load_trace_from_file(file_type: str, file_args: dict) -> MiniTrace:
    file_loader = {
        'HEKA DAT': MiniTrace.from_heka_file,
        'AXON ABF': MiniTrace.from_axon_file,
        'HDF5': MiniTrace.from_h5_file,
    }.get(file_type, None)

    if file_loader is None:
        raise ValueError('Unsupported file type.')

    return file_loader(**file_args)


def finalize_dialog_window(window: QDialog, title: str='new window', cancel: bool=True):
    QBtn = (QDialogButtonBox.Ok | QDialogButtonBox.Cancel) if cancel else QDialogButtonBox.Close
    window.buttonBox = QDialogButtonBox(QBtn)
    if cancel:
        window.buttonBox.accepted.connect(window.accept)
        window.buttonBox.rejected.connect(window.reject)
    else:
        window.buttonBox.clicked.connect(window.accept)
    window.layout.addRow(window.buttonBox)
    window.setWindowTitle(title)
    window.setWindowModality(pg.QtCore.Qt.ApplicationModal)


# ------- Classes ------- #
class minimlGuiMain(QMainWindow):
    def __init__(self):
        super(minimlGuiMain, self).__init__()
        self.initUI()
        self._create_toolbar()
        self._connect_actions()
        self.info_dialog = None
        self.settings = MinimlSettings()


    def initUI(self):
        self.statusbar = self.statusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.tracePlot = pg.PlotWidget()
        self.tracePlot.setBackground('w')
        self.tracePlot.showGrid(x=True, y=True)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Imon', '')

        self.eventPlot = pg.PlotWidget()
        self.eventPlot.setBackground('w')
        self.histogramPlot = pg.PlotWidget()
        self.histogramPlot.setBackground('w')
        self.averagePlot = pg.PlotWidget()
        self.averagePlot.setBackground('w')
        
        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter1.addWidget(self.eventPlot)
        self.splitter1.addWidget(self.histogramPlot)
        self.splitter1.addWidget(self.averagePlot)
        self.splitter1.setSizes([250,250,250])
        
        self.splitter2 = QSplitter(Qt.Vertical)
        self.splitter2.addWidget(self.tracePlot)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.setSizes([300,200])
        
        self.splitter3 = QSplitter(Qt.Horizontal)
        self.splitter3.addWidget(self.splitter2)

        self._create_table()

        self.splitter3.addWidget(self.tableWidget)
        self.splitter3.setSizes([780, 320])

        self.setCentralWidget(self.splitter3)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))
        
        self.setGeometry(150, 150, 1150, 650)
        self.setWindowTitle('miniML')
        self.show()
		

    def _create_toolbar(self):
        self.tb = self.addToolBar("Menu")
        self.tb.setMovable(False)

        self.openAction = QAction(QIcon("core/icons/load_file_24px_blue.svg"), "Open...", self)
        self.tb.addAction(self.openAction)
        self.filterAction = QAction(QIcon("core/icons/filter_24px_blue.svg"), "Filter", self)
        self.tb.addAction(self.filterAction)
        self.infoAction = QAction(QIcon("core/icons/info_24px_blue.svg"), "Info", self)
        self.tb.addAction(self.infoAction)
        self.cutAction = QAction(QIcon("core/icons/content_cut_24px_blue.svg"), "Cut trace", self)
        self.tb.addAction(self.cutAction)
        self.resetAction = QAction(QIcon("core/icons/restore_page_24px_blue.svg"), "Reload", self)
        self.tb.addAction(self.resetAction)
        self.analyseAction = QAction(QIcon("core/icons/rocket_launch_24px_blue.svg"), "Analyse", self)
        self.tb.addAction(self.analyseAction)
        self.plotAction = QAction(QIcon("core/icons/insert_chart_24px_blue.svg"), "Plot", self)
        self.tb.addAction(self.plotAction)
        self.tableAction = QAction(QIcon("core/icons/table_24px_blue.svg"), "Table", self)
        self.tb.addAction(self.tableAction)
        self.textsaveAction = QAction(QIcon("core/icons/textfile_24px_blue.svg"), "Save as TXT", self)
        self.tb.addAction(self.textsaveAction)
        self.saveAction = QAction(QIcon("core/icons/save_24px_blue.svg"), "Save as HDF5", self)
        self.tb.addAction(self.saveAction)
        self.settingsAction = QAction(QIcon("core/icons/settings_24px_blue.svg"), "Settings", self)
        self.tb.addAction(self.settingsAction)


    def _connect_actions(self):
        self.openAction.triggered.connect(self.new_file)
        self.filterAction.triggered.connect(self.filter_data)
        self.infoAction.triggered.connect(self.info_window)
        self.cutAction.triggered.connect(self.cut_data)
        self.resetAction.triggered.connect(self.reload_data)
        self.analyseAction.triggered.connect(self.run_analysis)
        self.plotAction.triggered.connect(self.toggle_plot_win)
        self.tableAction.triggered.connect(self.toggle_table_win)
        self.settingsAction.triggered.connect(self.settings_window)
        self.textsaveAction.triggered.connect(self.save_as_csv)
        self.saveAction.triggered.connect(self.save_as_hdf)


    def _create_table(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.verticalHeader().setDefaultSectionSize(10)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(90)
        self.tableWidget.setRowCount(0) 
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(("Position;Amplitude;Charge;Risetime;Decay").split(";"))
        self.tableWidget.viewport().installEventFilter(self)
        self.tableWidget.setSelectionBehavior(QTableView.SelectRows)


    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                index = self.tableWidget.indexAt(event.pos())   
                if index.data():
                    selected_ev = index.row()
            elif event.button() == Qt.RightButton:
                index = self.tableWidget.indexAt(event.pos())
                if index.isValid():
                    pass

        return super().eventFilter(source, event)


    def contextMenuEvent(self, event):
        gp = event.globalPos()
        vp_pos = self.tableWidget.viewport().mapFromGlobal(gp)
        row = self.tableWidget.rowAt(vp_pos.y())
        if row >= 0 and self.tableWidget.indexAt(vp_pos).data():
            self.menu = QMenu(self)
            inspectAction = QAction('Inspect event', self)
            inspectAction.triggered.connect(lambda: self.inspect_event(event, row))
            self.menu.addAction(inspectAction)

            deleteAction = QAction('Delete event', self)
            deleteAction.triggered.connect(lambda: self.delete_event(event, row))
            self.menu.addAction(deleteAction)

            self.menu.popup(QCursor.pos())


    def inspect_event(self, event, row):
        ''' zoom in onto selected event in main plot window '''
        xstart = int(self.detection.event_locations[row] - self.detection.window_size/2)
        xend = int(self.detection.event_locations[row] + self.detection.window_size)
        ymin = np.amin(self.detection.trace.data[xstart:xend]) * 1.05
        ymax = np.amax(self.detection.trace.data[xstart:xend]) * 1.05
        self.tracePlot.setXRange(xstart * self.detection.trace.sampling, xend * self.detection.trace.sampling)
        self.tracePlot.setYRange(ymin, ymax)


    def delete_event(self, event, row):
        msgbox = QMessageBox
        answer = msgbox.question(self,'', "Do you really want to delete this event?", msgbox.Yes | msgbox.No)

        if answer == msgbox.Yes:
            self.detection.event_locations = np.delete(self.detection.event_locations, row, axis=0)
            self.detection.event_peak_locations = np.delete(self.detection.event_peak_locations, row, axis=0)
            self.detection.event_peak_times = np.delete(self.detection.event_peak_times, row, axis=0)
            self.detection.event_peak_values = np.delete(self.detection.event_peak_values, row, axis=0)
            self.detection.event_start = np.delete(self.detection.event_start, row, axis=0)
            self.detection.decaytimes = np.delete(self.detection.decaytimes, row, axis=0)
            self.detection.risetimes = np.delete(self.detection.risetimes, row, axis=0)
            self.detection.charges = np.delete(self.detection.charges, row, axis=0)
            self.detection.event_bsls = np.delete(self.detection.event_bsls, row, axis=0)
            self.detection.events = np.delete(self.detection.events, row, axis=0)
            self.detection.event_scores = np.delete(self.detection.event_scores, row, axis=0)

            self.detection._eval_events(verbose=False)
            
            self.update_main_plot()
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[self.detection.event_peak_locations]
            pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
            self.plotDetected = self.tracePlot.plot(ev_positions, ev_peakvalues, pen=pen, symbol='o', symbolSize=8, 
                                                    symbolpen=self.settings.colors[0], symbolBrush=self.settings.colors[0])

            self.tabulate_results()
            self.plot_events()


    def filter_data(self):
        if not hasattr(self, 'trace'):
            return

        panel = FilterPanel(self)
        panel.exec_()
        if panel.result() == 0:
            return

        if panel.detrend.isChecked():
            self.trace = self.trace.filter(highpass=float(panel.high.text()), order=int(panel.order.text()))
        if panel.notch.isChecked():
            self.trace = self.trace.filter(notch=float(panel.notch_freq.text()))
        if panel.lowpass.isChecked():
            if panel.filter_type.currentText() == 'Chebyshev':
                self.trace = self.trace.filter(lowpass=float(panel.low.text()), order=int(panel.order.text()))
            else:
                self.trace = self.trace.filter(savgol=float(panel.window.text()), order=int(panel.order.text()))
        
        self.update_main_plot()

        
    def cut_data(self):
        if not hasattr(self, 'trace'):
            return
        cut_win = CutPanel(self)
        cut_win.exec_()
        if cut_win.result() == 0:
            return

        start_x = int(float(cut_win.start.text()) / self.trace.sampling)
        end_x = int(float(cut_win.end.text()) / self.trace.sampling)

        self.trace.data = self.trace.data[start_x:end_x]
        self.update_main_plot()


    def update_main_plot(self):
        self.tracePlot.clear()
        pen = pg.mkPen(color=self.settings.colors[3], width=1)
        self.plotData = self.tracePlot.plot(self.trace.time_axis, self.trace.data, pen=pen)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Imon', self.trace.y_unit)
    

    def toggle_table_win(self):
        if 0 in self.splitter3.sizes():
            self.splitter3.setSizes(self._store_size)
        else:
            self._store_size = self.splitter3.sizes()
            self.splitter3.setSizes([np.sum(self.splitter3.sizes()), 0])


    def toggle_plot_win(self):
        if 0 in self.splitter2.sizes():
            self.splitter2.setSizes(self._store_size)
        else:
            self._store_size = self.splitter2.sizes()
            self.splitter2.setSizes([np.sum(self.splitter2.sizes()), 0])    


    def reload_data(self):
        if not hasattr(self, 'filename'):
            return

        msgbox = QMessageBox
        answer = msgbox.question(self,'', "Do you want to reload data?", msgbox.Yes | msgbox.No)

        if answer == msgbox.Yes:
            self.trace = load_trace_from_file(self.filetype, self.load_args)
            self.update_main_plot()
            self.tableWidget.clear()
            self.eventPlot.clear()
            self.histogramPlot.clear()
            self.averagePlot.clear()


    def new_file(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open file', '', 'HDF, DAT, or ABF files (*.h5 *.hdf *.hdf5 *.dat *.abf)')[0]
        if self.filename == '':
            return

        # HDF file
        if self.filename.endswith('h5'):
            panel = LoadHdfPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return

            self.filetype = 'HDF5'
            self.load_args = {'filename': self.filename,
                              'tracename': panel.e1.text(),
                              'sampling': float(panel.e2.text()),
                              'scaling': float(panel.e3.text()), 
                              'unit': panel.e4.text()}

        # ABF file
        elif self.filename.endswith('abf'):
            panel = LoadAbfPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return

            self.filetype = 'AXON ABF'
            self.load_args = {'filepath': self.filename,
                              'channel': int(panel.e1.text()),
                              'scaling': float(panel.e2.text()), 
                              'unit': panel.e3.text() if (panel.e3.text() != '') else None}

        # DAT file
        elif self.filename.endswith('dat'):
            panel = LoadDatPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return
            
            self.filetype = 'HEKA DAT'
            series_no, rectype = panel.series.currentText().split(' - ')
            group_no, _ = panel.group.currentText().split(' - ')
            try:
                series_list = [int(s) for s in panel.e1.text().replace(',', ';').split(';')]
            except ValueError:
                series_list = None if panel.load_option.isChecked() else [int(series_no)]
            
            self.load_args = {'filename': self.filename,
                              'rectype': rectype,
                              'group': int(group_no),
                              'exclude_series': series_list,
                              'scaling': float(panel.e2.text()),
                              'unit': panel.e3.text() if (panel.e3.text() != '') else None}

        self.trace = load_trace_from_file(self.filetype, self.load_args)
        self.update_main_plot()
        self.tableWidget.setRowCount(0)
        self.eventPlot.clear()
        self.histogramPlot.clear()
        self.averagePlot.clear()
        self.detection = EventDetection(self.trace)
        
    
    def info_window(self):
        if not hasattr(self, 'trace'):
            return

        info_win = FileInfoPanel(self)
        info_win.exec_()
    

    def settings_window(self):
        settings_win = SettingsPanel(self)
        settings_win.exec_()
        if settings_win.result() == 0:
            return

        self.settings.stride = int(settings_win.stride.text())
        self.settings.event_window = int(settings_win.ev_len.text())
        self.settings.model_path = str(settings_win.model.currentText())
        self.settings.model_name = str(settings_win.model.currentText())
        self.settings.event_threshold = float(settings_win.thresh.text())
        self.settings.direction = str(settings_win.direction.currentText())
        self.settings.batch_size = int(settings_win.batchsize.text())


    def run_analysis(self):
        if not hasattr(self, 'trace'):
            return
        
        n_batches = np.ceil((self.trace.data.shape[0] - self.settings.event_window) / (self.settings.stride * self.settings.batch_size)).astype(int)
        n_batches = np.floor(n_batches/5)
        tf.get_logger().setLevel("ERROR")

        with pg.ProgressDialog('Detecting events', minimum=0, maximum=n_batches, busyCursor=True, cancelText=None) as self.dlg:
            def update_progress():
                self.dlg += 1
        
            class CustomCallback(tf.keras.callbacks.Callback):
                def on_predict_batch_end(self, batch, logs=None):
                    if batch % 5 == 0: 
                        update_progress()

            self.detection = EventDetection(data=self.trace,
                                            model_path=self.settings.model_path,
                                            model_threshold=self.settings.event_threshold,
                                            window_size=self.settings.event_window,
                                            batch_size=self.settings.batch_size,
                                            event_direction=self.settings.direction,
                                            verbose=0,
                                            callbacks=CustomCallback())

            self.detection.detect_events(stride=self.settings.stride, eval=True)

        if self.detection.event_locations.shape[0] > 0:
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[self.detection.event_peak_locations]
            pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
            self.plotDetected = self.tracePlot.plot(ev_positions, ev_peakvalues, pen=pen, symbol='o', symbolSize=8, 
                                                    symbolpen=self.settings.colors[0], symbolBrush=self.settings.colors[0])

            self.tabulate_results()
            self.plot_events()
        else:
            print('no events detected.')
            

    def save_as_csv(self):
        if not hasattr(self, 'detection'):
            return

        folder_path = QFileDialog.getExistingDirectory(self, 'Open Directory', '')
        if folder_path == '':
            return

        self.detection.save_to_csv(path=folder_path)


    def save_as_hdf(self):
        if not hasattr(self, 'detection'):
            return    

        file_name = QFileDialog.getSaveFileName(self, 'Save file', '', 'HDF files (*.h5 *.hdf *.hdf5)')[0]
        if file_name == '':
            return

        self.detection.save_to_h5(filename=file_name)


    def plot_events(self):
        self.eventPlot.clear()
        self.eventPlot.setTitle("Detected events")
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        for event in self.detection.events:
            pen = pg.mkPen(color=self.settings.colors[3], width=1)
            self.eventPlot.plot(time_data, event, pen=pen)
        self.eventPlot.setLabel('bottom', 'Time', 's')
        self.eventPlot.setLabel('left', 'Amplitude', 'pA')

        y, x = np.histogram(self.detection.event_stats.amplitudes, bins='auto')
        curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=self.settings.colors[3])
        self.histogramPlot.clear()
        self.histogramPlot.setTitle("Amplitude histogram")
        self.histogramPlot.addItem(curve)
        self.histogramPlot.setLabel('bottom', 'Amplitude', 'pA')
        self.histogramPlot.setLabel('left', 'Count', '')

        ev_average = np.mean(self.detection.events, axis=0)
        self.averagePlot.clear()
        self.averagePlot.setTitle("Average event waveform")
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        pen = pg.mkPen(color=self.settings.colors[0], width=2)
        self.averagePlot.plot(time_data, ev_average, pen=pen)
        self.averagePlot.setLabel('bottom', 'Time', 's')
        self.averagePlot.setLabel('left', 'Amplitude', 'pA')

    
    def tabulate_results(self):
        self.tableWidget.clear()
        n_events = len(self.detection.event_stats.amplitudes)
        self.tableWidget.setHorizontalHeaderLabels(['Location', 'Amplitude', 'Charge', 'Risetime', 'Decay'])
        self.tableWidget.setRowCount(n_events)
        for i in range(n_events):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(f'{self.detection.event_locations[i] * self.detection.trace.sampling :.5f}'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(f'{self.detection.event_stats.amplitudes[i]:.5f}'))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(f'{self.detection.event_stats.charges[i]:.5f}'))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(f'{self.detection.event_stats.risetimes[i]:.5f}'))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(f'{self.detection.event_stats.halfdecays[i]:.5f}'))
        self.tableWidget.show()


class LoadHdfPanel(QDialog):
    def __init__(self, parent=None):
        super(LoadHdfPanel, self).__init__(parent)
        
        self.e1 = QLineEdit('mini_data')
        self.e2 = QLineEdit('2e-5')
        self.e3 = QLineEdit('1e12')
        self.e4 = QLineEdit('pA')
        
        self.layout = QFormLayout(self)
        self.layout.addRow('Dataset name:', self.e1)
        self.layout.addRow('Sampling interval (s):', self.e2)
        self.layout.addRow('Scaling factor:', self.e3)
        self.layout.addRow('Data unit:', self.e4)

        finalize_dialog_window(self, title='Load HDF .h5 file')


class LoadAbfPanel(QDialog):
    def __init__(self, parent=None):
        super(LoadAbfPanel, self).__init__(parent)
        
        self.e1 = QLineEdit('0')
        self.e2 = QLineEdit('1')
        self.e3 = QLineEdit('')

        self.layout = QFormLayout(self)
        self.layout.addRow('Recording channel:', self.e1)
        self.layout.addRow('Scaling factor:', self.e2)
        self.layout.addRow('Data unit:', self.e3)

        finalize_dialog_window(self, title='Load AXON .abf file')


class LoadDatPanel(QDialog):
    def __init__(self, parent=None):
        super(LoadDatPanel, self).__init__(parent)
        
        self.bundle = heka.Bundle(parent.filename)

        group_series = []
        for i, GroupRecord in enumerate(self.bundle.pul.children):
            group_series.append(str(i + 1) + ' - ' + GroupRecord.Label)
        self.group = QComboBox()
        self.group.addItems(group_series)
        self.group.setMinimumWidth(150)
        self.group.currentIndexChanged[str].connect(self.on_comboBoxParent_currentIndexChanged)

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[0].children):
            bundle_series.append(str(i + 1) + ' - ' + SeriesRecord.Label)
        self.series = QComboBox()
        self.series.addItems(bundle_series)
        self.series.setMinimumWidth(300)
        self.load_option = QCheckBox('')
        self.e1 = QLineEdit('')
        self.e2 = QLineEdit('1e12')
        self.e3 = QLineEdit('pA')

        self.layout = QFormLayout(self)
        self.layout.addRow('Import group:', self.group)
        self.layout.addRow('Import series:', self.series)
        self.layout.addRow('Import all series of this type:', self.load_option)
        self.layout.addRow('Exclude selected series:', self.e1)
        self.layout.addRow('Scaling factor:', self.e2)
        self.layout.addRow('Data unit:', self.e3)

        finalize_dialog_window(self, title='Load HEKA .dat file')

    @pyqtSlot(str)
    def on_comboBoxParent_currentIndexChanged(self, index):
        group_no, _ = index.split(' - ')

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[int(group_no) - 1].children):
            bundle_series.append(str(i + 1) + ' - ' + SeriesRecord.Label)

        self.series.clear()
        self.series.addItems(bundle_series)



class FileInfoPanel(QDialog):
    def __init__(self, parent=None):
        super(FileInfoPanel, self).__init__(parent)

        self.filename = QLineEdit(parent.trace.filename)
        self.filename.setReadOnly(True)
        self.filename.setFixedWidth(300)
        self.format = QLineEdit(parent.filetype)
        self.format.setReadOnly(True)
        self.length = QLineEdit(f'{parent.trace.total_time:.2f}')
        self.length.setReadOnly(True)
        self.unit = QLineEdit(parent.trace.y_unit)
        self.unit.setReadOnly(True)
        self.sampling = QLineEdit(str(np.round(parent.trace.sampling_rate)))
        self.sampling.setReadOnly(True)
        
        self.layout = QFormLayout(self)
        self.layout.addRow('Filename:', self.filename)
        self.layout.addRow('File format:', self.format)
        self.layout.addRow('Recording duration (s):', self.length)    
        self.layout.addRow('Data unit', self.unit)
        self.layout.addRow('Sampling rate (Hz):', self.sampling)

        finalize_dialog_window(self, title='File info', cancel=False)


class SettingsPanel(QDialog):
    def __init__(self, parent=None):
        super(SettingsPanel, self).__init__(parent)
        
        self.stride = QLineEdit(str(parent.settings.stride))
        self.ev_len = QLineEdit(str(parent.settings.event_window))
        self.thresh = QLineEdit(str(parent.settings.event_threshold))
        self.model = QComboBox()
        self.model.addItems(get_available_models())
        index = self.model.findText(parent.settings.model_name)
        if index >= 0:
            self.model.setCurrentIndex(index)
        self.model.setFixedWidth(200)
        self.direction = QComboBox()
        self.direction.addItems(['negative', 'positive'])
        self.direction.setFixedWidth(200)
        self.batchsize = QLineEdit(str(parent.settings.batch_size))

        self.layout = QFormLayout(self)
        self.layout.addRow('Stride length (samples)', self.stride)
        self.layout.addRow('Event length (samples)', self.ev_len)
        self.layout.addRow('Detection threshold', self.thresh)
        self.layout.addRow('Model', self.model)
        self.layout.addRow('Event direction', self.direction)
        self.layout.addRow('Batch size', self.batchsize)

        finalize_dialog_window(self, title='miniML settings')


class CutPanel(QDialog):
    def __init__(self, parent=None):
        super(CutPanel, self).__init__(parent)
        
        self.start = QLineEdit('0.0')
        self.end = QLineEdit(str(np.round(parent.trace.total_time)))
        
        self.layout = QFormLayout(self)
        self.layout.addRow('new start (s)', self.start)
        self.layout.addRow('new end (s)', self.end)    

        finalize_dialog_window(self, title='Cut trace')


class FilterPanel(QDialog):
    def __init__(self, parent=None):
        super(FilterPanel, self).__init__(parent)
        
        def comboBoxIndexChanged(index):
            if index == 1:
                self.window.setEnabled(True)
                self.low.setEnabled(False)
            else:
                self.window.setEnabled(False)
                self.low.setEnabled(True)

        self.detrend = QCheckBox('')
        self.high = QLineEdit('0.5')
        self.high.setValidator(QDoubleValidator(0.01,99.99,2))
        self.notch = QCheckBox('')
        self.notch_freq = QLineEdit('50')
        self.notch_freq.setValidator(QDoubleValidator(0.9,9999.9,1))
        self.lowpass = QCheckBox('')
        self.low = QLineEdit('750')
        self.low.setValidator(QDoubleValidator(0.9,99999.9,1))
        self.filter_type = QComboBox()
        self.filter_type.addItems(['Chebyshev', 'Savitzky-Golay'])
        self.filter_type.currentIndexChanged.connect(comboBoxIndexChanged)
        self.filter_type.setFixedWidth(200)
        self.window = QLineEdit('5.0')
        self.window.setValidator(QDoubleValidator(0.001,999.9,3))
        self.window.setEnabled(False)
        self.order = QLineEdit('4')
        self.order.setValidator(QIntValidator(1,9))
        
        self.layout = QFormLayout(self)
        self.layout.addRow('Detrend data', self.detrend)
        self.layout.addRow('High-pass (Hz)', self.high)
        self.layout.addRow('Notch filter', self.notch)
        self.layout.addRow('Notch frequency (Hz)', self.notch_freq)        
        self.layout.addRow('Lowpass filter', self.lowpass)
        self.layout.addRow('Filter type', self.filter_type)
        self.layout.addRow('Low-pass (Hz)', self.low)
        self.layout.addRow('Window (ms)', self.window)
        self.layout.addRow('Filter order', self.order)

        finalize_dialog_window(self, title='Filter settings')



if __name__ == '__main__':

    pg.setConfigOption('leftButtonPan', False)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('core/minML_icon.png'))
    main = minimlGuiMain()
    app.setStyleSheet(build_stylesheet(theme='light_blue.xml', template='core/miniml.css.template'))
    main.show()
    sys.exit(app.exec_())
