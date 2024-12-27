# ------- Imports ------- #
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, QDialogButtonBox, QSplitter, QAction, 
                             QTableWidget, QTableView, QMenu, QStyleFactory, QMessageBox, QFileDialog,
                             QLineEdit, QFormLayout, QCheckBox, QTableWidgetItem, QComboBox, QLabel)
from PyQt5.QtCore import Qt, QEvent, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QCursor, QDoubleValidator, QIntValidator, QPixmap
import pyqtgraph as pg
import numpy as np
import tensorflow as tf
from pathlib import Path
import h5py
import pyabf
from qt_material import build_stylesheet
import sys
from miniML import MiniTrace, EventDetection
from miniML_settings import MinimlSettings
import FileImport.HekaReader as heka



# ------- GUI config ------- #
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('leftButtonPan', False)


# ------- Functions ------- #
def get_available_models() -> list:
    """
    Returns a list of available model paths in the /models folder.
    The list only contains relative paths.
    """
    models_dir = Path(__file__).parent.parent / 'models'
    models = [str(p.relative_to(models_dir)) for p in models_dir.glob('**/*.h5')]

    return models


def get_hdf_keys(filepath: str) -> list:
    """ 
    Returns a list of keys in an hdf5 file. 
    """
    with h5py.File(filepath, 'r') as f:
        return list(f.keys())


def load_trace_from_file(file_type: str, file_args: dict) -> MiniTrace:
    file_loader = {
        'HEKA DAT': MiniTrace.from_heka_file,
        'AXON ABF': MiniTrace.from_axon_file,
        'HDF5': MiniTrace.from_h5_file,
    }.get(file_type, None)

    if file_loader is None:
        raise ValueError('Unsupported file type.')

    return file_loader(**file_args)


def finalize_dialog_window(window: QDialog, title: str='new window', cancel: bool=True) -> None:
    """
    Finalizes a dialog window by adding a OK/Cancel button box to it and setting the window title.

    Args:
        window (QDialog): The dialog window to finalize.
        title (str, optional): The title of the window. Defaults to 'new window'.
        cancel (bool, optional): Whether to include a cancel button. Defaults to True.

    Returns:
        None
    """
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
        self._create_menubar()
        self.info_dialog = None
        self.settings = MinimlSettings()


    def initUI(self):
        self.statusbar = self.statusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.tracePlot = pg.PlotWidget()
        # self.tracePlot.showGrid(x=True, y=True)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Imon', '')
        
        self.predictionPlot = pg.PlotWidget()
        # self.predictionPlot.showGrid(x=True, y=True)
        self.predictionPlot.setLabel('left', 'Confidence', '')
        self.predictionPlot.setXLink(self.tracePlot)

        self.eventPlot = pg.PlotWidget()
        self.histogramPlot = pg.PlotWidget()
        self.averagePlot = pg.PlotWidget()
        
        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter1.setHandleWidth(12)
        self.splitter1.addWidget(self.eventPlot)
        self.splitter1.addWidget(self.averagePlot)
        self.splitter1.addWidget(self.histogramPlot)
        self.splitter1.setSizes([250,250,250])
        
        self.splitter2 = QSplitter(Qt.Vertical)
        self.splitter2.setHandleWidth(12)
        self.splitter2.addWidget(self.predictionPlot)
        self.splitter2.addWidget(self.tracePlot)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.setSizes([130,270,150])
        
        self.splitter3 = QSplitter(Qt.Horizontal)
        self.splitter3.setHandleWidth(12)
        self.splitter3.addWidget(self.splitter2)

        self._create_table()

        self.splitter3.addWidget(self.tableWidget)
        self.splitter3.setSizes([750, 400])

        self.setCentralWidget(self.splitter3)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))
        
        self.setGeometry(100, 100, 1150, 750)
        self.setWindowTitle('miniML')
        self.show()
	

    def _create_menubar(self):
        menubar = self.menuBar()

        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.resetAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.closeAction)

        editMenu = menubar.addMenu('Edit')
        editMenu.addAction(self.filterAction)
        editMenu.addAction(self.cutAction)
        editMenu.addAction(self.infoAction)

        viewMenu = menubar.addMenu('View')
        viewMenu.addAction(self.plotAction)
        viewMenu.addAction(self.tableAction)
        viewMenu.addAction(self.predictionAction)

        runMenu = menubar.addMenu('Run')
        runMenu.addAction(self.settingsAction)
        runMenu.addAction(self.analyseAction)
        runMenu.addAction(self.summaryAction)

        helpMenu = menubar.addMenu('Help')
        helpMenu.addAction(self.aboutAction)


    def _create_toolbar(self):
        self.tb = self.addToolBar('Menu')
        self.tb.setMovable(False)

        self.openAction = QAction(QIcon('icons/load_file_24px_blue.svg'), 'Open...', self)
        self.openAction.setShortcut('Ctrl+O')
        self.tb.addAction(self.openAction)
        self.filterAction = QAction(QIcon('icons/filter_24px_blue.svg'), 'Filter', self)
        self.filterAction.setShortcut('Ctrl+F')
        self.tb.addAction(self.filterAction)
        self.infoAction = QAction(QIcon('icons/troubleshoot_24px_blue.svg'), 'Info', self)
        self.infoAction.setShortcut('Ctrl+I')
        self.tb.addAction(self.infoAction)
        self.cutAction = QAction(QIcon('icons/content_cut_24px_blue.svg'), 'Cut trace', self)
        self.cutAction.setShortcut('Ctrl+X')
        self.tb.addAction(self.cutAction)
        self.resetAction = QAction(QIcon('icons/restore_page_24px_blue.svg'), 'Reload', self)
        self.resetAction.setShortcut('Ctrl+R')
        self.tb.addAction(self.resetAction)
        self.analyseAction = QAction(QIcon('icons/rocket_launch_24px_blue.svg'), 'Analyse', self)
        self.analyseAction.setShortcut('Ctrl+A')
        self.tb.addAction(self.analyseAction)
        self.predictionAction = QAction(QIcon('icons/ssid_chart_24px_blue.svg'), 'Prediction', self)
        self.tb.addAction(self.predictionAction)
        self.summaryAction = QAction(QIcon('icons/functions_24px_blue.svg'), 'Summary', self)
        self.tb.addAction(self.summaryAction)
        self.plotAction = QAction(QIcon('icons/insert_chart_24px_blue.svg'), 'Plot', self)
        self.tb.addAction(self.plotAction)
        self.tableAction = QAction(QIcon('icons/table_24px_blue.svg'), 'Table', self)
        self.tb.addAction(self.tableAction)
        self.saveAction = QAction(QIcon('icons/save_24px_blue.svg'), 'Save results', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.tb.addAction(self.saveAction)
        self.settingsAction = QAction(QIcon('icons/settings_24px_blue.svg'), 'Settings', self)
        self.settingsAction.setShortcut('Ctrl+P')
        self.tb.addAction(self.settingsAction)
        self.closeAction = QAction(QIcon('icons/cancel_24px_blue'), 'Close Window', self)
        self.closeAction.setShortcut('Ctrl+W')
        # self.tb.addAction(self.closeAction)
        self.aboutAction = QAction(QIcon('icons/info_24px_blue.svg'), 'About', self)
        self.aboutAction.setShortcut('Ctrl+H')
        # self.tb.addAction(self.aboutAction)
        

    def _connect_actions(self):
        self.openAction.triggered.connect(self.new_file)
        self.filterAction.triggered.connect(self.filter_data)
        self.infoAction.triggered.connect(self.info_window)
        self.cutAction.triggered.connect(self.cut_data)
        self.resetAction.triggered.connect(self.reload_data)
        self.analyseAction.triggered.connect(self.run_analysis)
        self.predictionAction.triggered.connect(self.toggle_prediction_win)
        self.summaryAction.triggered.connect(self.summary_window)
        self.plotAction.triggered.connect(self.toggle_plot_win)
        self.tableAction.triggered.connect(self.toggle_table_win)
        self.settingsAction.triggered.connect(self.settings_window)
        self.saveAction.triggered.connect(self.save_results)
        self.closeAction.triggered.connect(self.close_gui)
        self.aboutAction.triggered.connect(self.about_win)


    def _create_table(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.verticalHeader().setDefaultSectionSize(10)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(90)
        self.tableWidget.setRowCount(0) 
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['Position', 'Amplitude', 'Area', 'Risetime', 'Decay'])
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


    def contextMenuEvent(self, event) -> None:
        """
        Create a context menu for the selected event.
        """
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


    def inspect_event(self, event, row) -> None:
        """
        Zoom in onto the selected event in main plot window. 
        """
        xstart = int(self.detection.event_locations[row] - self.detection.window_size/2)
        xend = int(self.detection.event_locations[row] + self.detection.window_size)
        ymin = np.amin(self.detection.trace.data[xstart:xend]) * 1.05
        ymax = np.amax(self.detection.trace.data[xstart:xend]) * 1.05
        self.tracePlot.setXRange(xstart * self.detection.trace.sampling, xend * self.detection.trace.sampling)
        self.tracePlot.setYRange(ymin, ymax)


    def delete_event(self, event, row) -> None:
        """
        Deletes an event from the detection object.

        Args:
            event (QEvent): The event that triggered the deletion.
            row (int): The index of the event to be deleted.

        Returns:
            None

        This function prompts the user with a confirmation dialog to delete an event. 
        After deleting the event, the function updates the main plot, plots the detected events, and tabulates the results.
        """
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

            self.detection._eval_events()
            
            self.update_main_plot()
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[self.detection.event_peak_locations]
            pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
            self.plotDetected = self.tracePlot.plot(ev_positions, ev_peakvalues, pen=pen, symbol='o', symbolSize=8, 
                                                    symbolpen=self.settings.colors[0], symbolBrush=self.settings.colors[0])

            self.tabulate_results()
            self.plot_events()


    def filter_data(self) -> None:
        """
        A function that filters data based on the selected filter options in the FilterPanel.
        Otherwise, it applies various filters (detrend, highpass, notch, lowpass) to the trace data based on the user-selected options in the FilterPanel.
        The function then updates the main plot with the filtered data.
        """

        if not hasattr(self, 'trace'):
            return

        panel = FilterPanel(self)
        panel.exec_()
        if panel.result() == 0:
            return

        if panel.detrend.isChecked():
            self.trace = self.trace.detrend(num_segments = int(panel.num_segments.text()))
        if panel.highpass.isChecked():
            self.trace = self.trace.filter(highpass=float(panel.high.text()), order=int(panel.order.text()))
        if panel.notch.isChecked():
            self.trace = self.trace.filter(notch=float(panel.notch_freq.text()))
        if panel.lowpass.isChecked():
            if panel.filter_type.currentText() == 'Chebyshev':
                self.trace = self.trace.filter(lowpass=float(panel.low.text()), order=int(panel.order.text()))
            else:
                self.trace = self.trace.filter(savgol=float(panel.window.text()), order=int(panel.order.text()))
        if panel.hann.isChecked():
                self.trace = self.trace.filter(hann=int(panel.hann_window.text()))

        self.update_main_plot()

        
    def cut_data(self) -> None:
        """
        Display the CutPanel window for slicing the data trace.
        """
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


    def update_main_plot(self) -> None:
        """
        Updates the main plot with the data trace.
        """
        self.tracePlot.clear()
        pen = pg.mkPen(color=self.settings.colors[3], width=1)
        self.plotData = self.tracePlot.plot(self.trace.time_axis, self.trace.data, pen=pen)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        label1 = 'Vmon' if self.recording_mode == 'current-clamp' else 'Imon'
        self.tracePlot.setLabel('left', label1, self.trace.y_unit)
    

    def toggle_table_win(self) -> None:
        if 0 in self.splitter3.sizes():
            self.splitter3.setSizes(self._store_size_c)
        else:
            self._store_size_c = self.splitter3.sizes()
            self.splitter3.setSizes([np.sum(self.splitter3.sizes()), 0])


    def toggle_plot_win(self) -> None:
        sizes = self.splitter2.sizes()
        if sizes[2] == 0: # panel is hidden
            sizes[0] = 0 if sizes[0] == 0 else self._store_size[0]
            sizes[1] = (np.sum(sizes[0:3]) - self._store_size_b) if sizes[0] == 0 else (self._store_size[1] - self._store_size_b)
            sizes[2] = self._store_size_b
            self._store_size = sizes
        else: # panel is shown
            self._store_size = sizes
            self._store_size_b = sizes[2]
            sizes[0] = 0 if sizes[0] == 0 else sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[0] == 0 else np.sum(sizes[1:3])
            sizes[2] = 0
        self.splitter2.setSizes(sizes)


    def toggle_prediction_win(self) -> None:
        sizes = self.splitter2.sizes()
        if sizes[0] == 0: # panel is hidden
            sizes[0] = self._store_size_a
            sizes[1] = (np.sum(sizes[0:3]) - self._store_size_a) if sizes[2] == 0 else (self._store_size[1] - self._store_size_a)
            sizes[2] = 0 if sizes[2] == 0 else self._store_size[2]
            self._store_size = sizes
        else: # panel is shown
            self._store_size = sizes
            self._store_size_a = sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[2] == 0 else np.sum(sizes[0:2])
            sizes[2] = 0 if sizes[2] == 0 else sizes[2]
            sizes[0] = 0
        self.splitter2.setSizes(sizes)


    def reload_data(self) -> None:
        """
        Reload the data from file and reset all windows.
        """
        if not hasattr(self, 'filename'):
            return

        msgbox = QMessageBox
        answer = msgbox.question(self,'', 'Do you want to reload data?', msgbox.Yes | msgbox.No)

        if answer == msgbox.Yes:
            self.trace = load_trace_from_file(self.filetype, self.load_args)
            self.update_main_plot()
            self.reset_windows()
            self.was_analyzed = False
            self.detection = EventDetection(self.trace)


    def reset_windows(self) -> None:
        """
        Clear all plot and table windows.
        """
        self.tableWidget.setRowCount(0)
        self.eventPlot.clear()
        self.histogramPlot.clear()
        self.averagePlot.clear()
        self.predictionPlot.clear()


    def close_gui(self) -> None:
        self.close()
    

    def about_win(self) -> None:
        """
        Display the About window.
        """
        about = AboutPanel(self)
        about.exec_()


    def new_file(self) -> None:
        """
        Open a new file via OS dialog and load data from it. Plots the data and initiates a detection object.
        """
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
            self.protocol = 'none'
            self.load_args = {'filename': self.filename,
                              'tracename': panel.e1.currentText(),
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
            self.protocol = panel.protocol.text()
            self.load_args = {'filename': self.filename,
                              'channel': int(panel.channel.currentText()),
                              'scaling': float(panel.scale.text()), 
                              'unit': panel.unit.text() if (panel.unit.text() != '') else None}

        # DAT file
        elif self.filename.endswith('dat'):
            panel = LoadDatPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return
            
            self.filetype = 'HEKA DAT'
            series_no, rectype = panel.series.currentText().split(' - ')
            self.protocol = rectype
            group_no, _ = panel.group.currentText().split(' - ')
            try:
                series_list = [int(s) for s in panel.e1.text().replace(',', ';').split(';')]
            except ValueError:
                series_list = [] if panel.load_option.isChecked() else [int(series_no)]
            
            self.load_args = {'filename': self.filename,
                              'rectype': rectype,
                              'group': int(group_no),
                              'exclude_series': series_list,
                              'scaling': float(panel.e2.text()),
                              'unit': panel.e3.text() if (panel.e3.text() != '') else None}
            
        self.trace = load_trace_from_file(self.filetype, self.load_args)
        self.recording_mode = 'current-clamp' if 'V' in self.trace.y_unit else 'voltage-clamp'
        
        self.update_main_plot()
        self.reset_windows()
        self.was_analyzed = False
        self.detection = EventDetection(self.trace)
        
    
    def info_window(self) -> None:
        if not hasattr(self, 'trace'):
            return

        info_win = FileInfoPanel(self)
        info_win.exec_()
    

    def summary_window(self) -> None:
        if not hasattr(self, 'trace'):
            return

        summary_win = SummaryPanel(self)
        summary_win.exec_()


    def settings_window(self) -> None:
        settings_win = SettingsPanel(self)
        settings_win.exec_()
        if settings_win.result() == 0:
            return

        self.settings.event_window = int(settings_win.ev_len.text())
        self.settings.stride = int(settings_win.stride.text())
        self.settings.model_path = str(settings_win.model.currentText())
        self.settings.model_name = str(settings_win.model.currentText())
        self.settings.event_threshold = float(settings_win.thresh.text()) if settings_win.thresh.hasAcceptableInput() else 0.5
        self.settings.direction = str(settings_win.direction.currentText())
        self.settings.batch_size = int(settings_win.batchsize.text())


    def run_analysis(self) -> None:
        """
        Run the miniML analysis on the loaded trace.
        """
        if not hasattr(self, 'trace'):
            return
        
        if self.was_analyzed:
            msgbox = QMessageBox
            answer = msgbox.question(self,'', 'Do you want to reanalyze this trace?', msgbox.Yes | msgbox.No)

            if answer == msgbox.No:
                return

            self.predictionPlot.clear()
            self.eventPlot.clear()
            self.averagePlot.clear()
            self.histogramPlot.clear()
            self.tableWidget.clear()
            self.update_main_plot()

        n_batches = np.ceil((self.trace.data.shape[0] - self.settings.event_window) / (self.settings.stride * self.settings.batch_size)).astype(int)
        n_batches = np.floor(n_batches/5)
        tf.get_logger().setLevel('ERROR')

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

            self.was_analyzed = True
            self.predictionPlot.clear()
            pen = pg.mkPen(color=self.settings.colors[3], width=1)
            prediction_x = np.arange(0, len(self.detection.prediction)) * self.trace.sampling
            self.predictionPlot.plot(prediction_x, self.detection.prediction, pen=pen)
            self.predictionPlot.plot([0, prediction_x[-1]], [self.settings.event_threshold, self.settings.event_threshold], 
                                     pen=pg.mkPen(color=self.settings.colors[0], style=Qt.DashLine, width=1))

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
            

    def save_results(self) -> None:
        if not hasattr(self, 'detection'):
            return
        default_filename = Path(self.filename).with_suffix('')
        file_types = 'CSV (*.csv);;Pickle (*.pickle);;HDF (*.h5 *.hdf *.hdf5)'
        save_filename, selected_filter = QFileDialog.getSaveFileName(self, 'Save file', str(default_filename), file_types)
        
        if not save_filename:
            return
        
        if selected_filter == 'CSV (*.csv)':
            self.detection.save_to_csv(filename=save_filename)
        elif selected_filter == 'Pickle (*.pickle)':
            self.detection.save_to_pickle(filename=save_filename)
        elif selected_filter == 'HDF (*.h5 *.hdf *.hdf5)':
            self.detection.save_to_h5(filename=save_filename)


    def plot_events(self):
        self.eventPlot.clear()
        self.eventPlot.setTitle('Detected events')
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        for event in self.detection.events:
            pen = pg.mkPen(color=self.settings.colors[3], width=1)
            self.eventPlot.plot(time_data, event, pen=pen)
        self.eventPlot.setLabel('bottom', 'Time', 's')
        self.eventPlot.setLabel('left', 'Amplitude', 'pA')

        y, x = np.histogram(self.detection.event_stats.amplitudes, bins='auto')
        curve = pg.PlotCurveItem(x, y, stepMode='center', fillLevel=0, brush=self.settings.colors[3])
        self.histogramPlot.clear()
        self.histogramPlot.setTitle('Amplitude histogram')
        self.histogramPlot.addItem(curve)
        self.histogramPlot.setLabel('bottom', 'Amplitude', 'pA')
        self.histogramPlot.setLabel('left', 'Count', '')

        ev_average = np.mean(self.detection.events, axis=0)
        self.averagePlot.clear()
        self.averagePlot.setTitle('Average event waveform')
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        pen = pg.mkPen(color=self.settings.colors[0], width=2)
        self.averagePlot.plot(time_data, ev_average, pen=pen)
        self.averagePlot.setLabel('bottom', 'Time', 's')
        self.averagePlot.setLabel('left', 'Amplitude', 'pA')

    
    def tabulate_results(self):
        self.tableWidget.clear()
        n_events = len(self.detection.event_stats.amplitudes)
        self.tableWidget.setHorizontalHeaderLabels(['Location', 'Amplitude', 'Area', 'Risetime', 'Decay'])
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
        
        self.e1 = QComboBox()
        self.e1.setMinimumWidth(200)
        self.e1.addItems(get_hdf_keys(parent.filename))
        self.e2 = QLineEdit('2e-5')
        self.e2.setMinimumWidth(200)
        self.e3 = QLineEdit('1e12')
        self.e3.setMinimumWidth(200)
        self.e4 = QLineEdit('pA')
        self.e4.setMinimumWidth(200)
        
        self.layout = QFormLayout(self)
        self.layout.addRow('Dataset name:', self.e1)
        self.layout.addRow('Sampling interval (s):', self.e2)
        self.layout.addRow('Scaling factor:', self.e3)
        self.layout.addRow('Data unit:', self.e4)

        finalize_dialog_window(self, title='Load HDF .h5 file')


class LoadAbfPanel(QDialog):
    def __init__(self, parent=None):
        super(LoadAbfPanel, self).__init__(parent)
        
        self.abf_file = pyabf.ABF(parent.filename)

        self.channel = QComboBox()
        self.channel.addItems([str(channel) for channel in self.abf_file.channelList])
        self.channel.setMinimumWidth(150)
        self.channel.currentIndexChanged[str].connect(self.on_comboBoxParent_currentChannelChanged)

        self.scale = QLineEdit('1')
        self.unit = QLineEdit(self.abf_file.adcUnits[0])
        self.protocol = QLineEdit(self.abf_file.protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setMinimumWidth(300)

        self.layout = QFormLayout(self)
        self.layout.addRow('Recording channel:', self.channel)
        self.layout.addRow('Scaling factor:', self.scale)
        self.layout.addRow('Data unit:', self.unit)
        self.layout.addRow('Protocol:', self.protocol)

        finalize_dialog_window(self, title='Load AXON .abf file')


    @pyqtSlot(str)
    def on_comboBoxParent_currentChannelChanged(self, index):

        self.unit.clear()
        self.unit.setText(self.abf_file.adcUnits[int(index)])


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
        self.mode = QLineEdit(parent.recording_mode)
        self.mode.setReadOnly(True)
        self.sampling = QLineEdit(str(np.round(parent.trace.sampling_rate)))
        self.sampling.setReadOnly(True)
        self.protocol = QLineEdit(parent.protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setFixedWidth(300)
        
        self.layout = QFormLayout(self)
        self.layout.addRow('Filename:', self.filename)
        self.layout.addRow('File format:', self.format)
        self.layout.addRow('Recording duration (s):', self.length)
        self.layout.addRow('Data unit', self.unit)
        self.layout.addRow('Recording mode:', self.mode)
        self.layout.addRow('Sampling rate (Hz):', self.sampling)
        self.layout.addRow('Protocol:', self.protocol)

        finalize_dialog_window(self, title='File info', cancel=False)


class AboutPanel(QDialog):
    def __init__(self, parent=None):
        super(AboutPanel, self).__init__(parent)

        self.layout = QFormLayout(self)

        logo = QLabel()
        logo.setPixmap(QPixmap(str(Path(__file__).parent.parent / 'minML_icon.png')).scaled(QSize(100, 100)))
        self.layout.addRow(logo)

        self.version = QLabel('miniML version 1.0.0')
        self.layout.addRow(self.version)

        self.author = QLabel('Authors: Philipp O\'Neill, Martin Baccino Calace, Igor Delvendahl')
        self.layout.addRow(self.author)

        self.website = QLabel('Website: <a href=\"https://github.com/delvendahl/miniML\">miniML GitHub repository</a>')
        self.website.setOpenExternalLinks(True)
        self.layout.addRow(self.website)

        self.paper = QLabel('Publication: <a href=\"https://doi.org/10.7554/eLife.98485.1\">miniML eLife paper 2024</a>')
        self.paper.setOpenExternalLinks(True)
        self.layout.addRow(self.paper)

        finalize_dialog_window(self, title='About miniML', cancel=False)



class SummaryPanel(QDialog):
    def __init__(self, parent=None):
        super(SummaryPanel, self).__init__(parent)

        self.populate_fields(parent)
        self.layout.addRow('Filename:', self.filename)
        self.layout.addRow('Events found:', self.event_count)
        self.layout.addRow('Event frequency (Hz):', self.event_frequency)
        self.layout.addRow('Average score:', self.average_score)
        self.layout.addRow(f'Average amplitude ({parent.detection.trace.y_unit}):', self.average_amplitude)
        self.layout.addRow(f'Median amplitude ({parent.detection.trace.y_unit}):', self.median_amplitude)
        self.layout.addRow('Coefficient of variation:', self.amplitude_cv)
        self.layout.addRow(f'Average area ({parent.detection.trace.y_unit}*s):', self.average_area)
        self.layout.addRow('Average risetime (ms):', self.average_rise_time)
        self.layout.addRow(f'Average 50% decay time (ms):', self.average_decay_time)
        self.layout.addRow('Decay time constant (ms):', self.decay_tau)

        finalize_dialog_window(self, title='Summary', cancel=False)


    def populate_fields(self, parent):
        self.filename = QLineEdit(parent.trace.filename)
        self.filename.setReadOnly(True)
        self.event_count = QLineEdit(str(parent.detection.event_stats.event_count))
        self.event_count.setReadOnly(True)
        self.event_frequency = QLineEdit(f'{parent.detection.event_stats.frequency():.5f}')
        self.event_frequency.setReadOnly(True)
        self.average_score = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.event_scores):.5f}')
        self.average_score.setReadOnly(True)
        self.average_amplitude = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.amplitudes):.5f}')
        self.average_amplitude.setReadOnly(True)
        self.median_amplitude = QLineEdit(f'{parent.detection.event_stats.median(parent.detection.event_stats.amplitudes):.5f}')
        self.median_amplitude.setReadOnly(True)
        self.amplitude_cv = QLineEdit(f'{parent.detection.event_stats.cv(parent.detection.event_stats.amplitudes):.5f}')
        self.amplitude_cv.setReadOnly(True)
        self.average_area = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.charges):.5f}')
        self.average_area.setReadOnly(True)
        self.average_rise_time = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.risetimes)*1000:.5f}')
        self.average_rise_time.setReadOnly(True)
        self.average_decay_time = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.halfdecays)*1000:.5f}')
        self.average_decay_time.setReadOnly(True)
        self.decay_tau = QLineEdit(f'{parent.detection.event_stats.mean(parent.detection.event_stats.avg_tau_decay)*1000:.5f}')
        self.decay_tau.setReadOnly(True)
        self.layout = QFormLayout(self)


class SettingsPanel(QDialog):
    def __init__(self, parent=None):
        super(SettingsPanel, self).__init__(parent)
        
        self.stride = QLineEdit(str(parent.settings.stride))
        self.ev_len = QLineEdit(str(parent.settings.event_window))
        self.thresh = QLineEdit(str(parent.settings.event_threshold))
        validator = QDoubleValidator(0.0, 1.0, 3)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.thresh.setValidator(validator)

        self.model = QComboBox()
        self.model.addItems(get_available_models())
        index = self.model.findText(parent.settings.model_name)
        if index >= 0:
            self.model.setCurrentIndex(index)
        self.model.setFixedWidth(200)
        self.direction = QComboBox()
        self.direction.addItems(['negative', 'positive'])
        if parent.settings.direction == 'negative':
            self.direction.setCurrentIndex(0)
        else:
            self.direction.setCurrentIndex(1)
        self.direction.setFixedWidth(200)
        self.batchsize = QLineEdit(str(parent.settings.batch_size))

        self.layout = QFormLayout(self)
        self.layout.addRow('Stride length (samples)', self.stride)
        self.layout.addRow('Event length (samples)', self.ev_len)
        self.layout.addRow('Min. peak height (0-1)', self.thresh)
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
        self.num_segments = QLineEdit('1')
        self.num_segments.setValidator(QIntValidator(1,99))
        self.highpass = QCheckBox('')
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
        self.hann = QCheckBox('')
        self.hann_window = QLineEdit('20')
        self.hann_window.setValidator(QIntValidator(3,1000))


        self.layout = QFormLayout(self)
        self.layout.addRow('Detrend data', self.detrend)
        self.layout.addRow('Number of segments', self.num_segments)
        self.layout.addRow('High-pass filter', self.highpass)
        self.layout.addRow('High-pass (Hz)', self.high)
        self.layout.addRow('Notch filter', self.notch)
        self.layout.addRow('Notch frequency (Hz)', self.notch_freq)        
        self.layout.addRow('Lowpass filter', self.lowpass)
        self.layout.addRow('Filter type', self.filter_type)
        self.layout.addRow('Low-pass (Hz)', self.low)
        self.layout.addRow('Window (ms)', self.window)
        self.layout.addRow('Filter order', self.order)
        self.layout.addRow('Hann filter', self.hann)
        self.layout.addRow('Hann window size', self.hann_window)

        finalize_dialog_window(self, title='Filter settings')



if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(__file__).parent.parent / 'minML_icon.png')))
    main = minimlGuiMain()
    extra = {'density_scale': '-1',}
    app.setStyleSheet(build_stylesheet(theme='light_blue.xml', invert_secondary=False, 
                                       extra=extra, template='miniml.css.template'))
    main.show()
    sys.exit(app.exec_())
