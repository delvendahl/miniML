# ------- Imports ------- #
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, QDialogButtonBox, QSplitter, QAction, 
                             QTableWidget, QTableView, QMenu, QStyleFactory, QMessageBox, QFileDialog, QGridLayout,
                             QLineEdit, QFormLayout, QCheckBox, QTableWidgetItem, QComboBox, QLabel, QToolBar)
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
        self.was_analyzed = False


    def initUI(self):
        self.statusbar = self.statusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.tracePlot = pg.PlotWidget()
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Imon', '')
        
        self.predictionPlot = pg.PlotWidget()
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

        self.tableWidget = self._create_table()

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
        viewMenu.addAction(self.eventViewerAction)


    def _create_toolbar(self):
        self.tb = self.addToolBar('Menu')
        self.tb.setMovable(False)

        self.openAction = QAction(QIcon('icons/load_file_24px_blue.svg'), 'Open...', self)
        self.openAction.setShortcut('Ctrl+O')
        self.tb.addAction(self.openAction)
        self.filterAction = QAction(QIcon('icons/filter_24px_blue.svg'), 'Filter', self)
        self.filterAction.setShortcut('Ctrl+F')
        self.tb.addAction(self.filterAction)
        self.infoAction = QAction(QIcon('icons/info_24px_blue.svg'), 'Info', self)
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
        self.eventViewerAction = QAction(QIcon('icons/event_mode_24px_blue.svg'), 'Event Viewer', self)
        self.eventViewerAction.setShortcut('Ctrl+E')
        self.tb.addAction(self.eventViewerAction)
        self.saveAction = QAction(QIcon('icons/save_24px_blue.svg'), 'Save results', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.tb.addAction(self.saveAction)
        self.settingsAction = QAction(QIcon('icons/settings_24px_blue.svg'), 'Settings', self)
        self.settingsAction.setShortcut('Ctrl+P')
        self.tb.addAction(self.settingsAction)
        self.closeAction = QAction(QIcon('icons/cancel_24px_blue.svg'), 'Close Window', self)
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
        self.eventViewerAction.triggered.connect(self.show_event_viewer)


    def _create_table(self):
        tableWidget = QTableWidget()
        tableWidget.verticalHeader().setDefaultSectionSize(10)
        tableWidget.horizontalHeader().setDefaultSectionSize(90)
        tableWidget.setRowCount(0) 
        tableWidget.setColumnCount(5)
        tableWidget.setHorizontalHeaderLabels(['Position', 'Amplitude', 'Area', 'Risetime', 'Decay'])
        tableWidget.viewport().installEventFilter(self)
        tableWidget.setSelectionBehavior(QTableView.SelectRows)
        return tableWidget


    def _warning_box(self, message):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('Message')
        msgbox.setText(message)
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
 

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
            self.detection.bsl_starts = np.delete(self.detection.bsl_starts, row, axis=0)
            self.detection.bsl_ends = np.delete(self.detection.bsl_ends, row, axis=0)
            self.detection.min_positions_rise = np.delete(self.detection.min_positions_rise, row, axis=0)
            self.detection.max_positions_rise = np.delete(self.detection.max_positions_rise, row, axis=0)
            self.detection.half_decay = np.delete(self.detection.half_decay, row, axis=0)
            self.detection.events = np.delete(self.detection.events, row, axis=0)
            self.detection.event_scores = np.delete(self.detection.event_scores, row, axis=0)

            self.exclude_events = np.delete(self.exclude_events, row, axis=0)
            self.use_for_avg = np.delete(self.use_for_avg, row, axis=0)
            self.detection.singular_event_indices = np.where(self.use_for_avg == 1)[0]
            self.detection._eval_events()
            
            self.update_main_plot()

            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()


    def delete_multiple_events(self, rows:list=[]) -> None:
        """
        Deletes multiple events from the detection object after exclusion in the Event Viewer.

        Args:
            rows (list): list of the event indices to be deleted.

        Returns:
            None

        This function prompts the user with a confirmation dialog to delete the events. 
        After deleting the event, the function updates the main plot, plots the detected events, and tabulates the results.
        """        
        if len(rows) > 0:
            msgbox = QMessageBox
            answer = msgbox.question(self,'', f"Do you really want to delete {len(rows)} event(s)? This can not be reverted", msgbox.Yes | msgbox.No)

            if answer == msgbox.Yes:
                self.detection.event_locations = np.delete(self.detection.event_locations, rows, axis=0)
                self.detection.event_peak_locations = np.delete(self.detection.event_peak_locations, rows, axis=0)
                self.detection.event_peak_times = np.delete(self.detection.event_peak_times, rows, axis=0)
                self.detection.event_peak_values = np.delete(self.detection.event_peak_values, rows, axis=0)
                self.detection.event_start = np.delete(self.detection.event_start, rows, axis=0)
                self.detection.decaytimes = np.delete(self.detection.decaytimes, rows, axis=0)
                self.detection.risetimes = np.delete(self.detection.risetimes, rows, axis=0)
                self.detection.charges = np.delete(self.detection.charges, rows, axis=0)
                self.detection.event_bsls = np.delete(self.detection.event_bsls, rows, axis=0)
                self.detection.bsl_starts = np.delete(self.detection.bsl_starts, rows, axis=0)
                self.detection.bsl_ends = np.delete(self.detection.bsl_ends, rows, axis=0)
                self.detection.min_positions_rise = np.delete(self.detection.min_positions_rise, rows, axis=0)
                self.detection.max_positions_rise = np.delete(self.detection.max_positions_rise, rows, axis=0)
                self.detection.min_values_rise = np.delete(self.detection.min_values_rise, rows, axis=0)
                self.detection.max_values_rise = np.delete(self.detection.max_values_rise, rows, axis=0)                
                self.detection.half_decay = np.delete(self.detection.half_decay, rows, axis=0)
                self.detection.events = np.delete(self.detection.events, rows, axis=0)
                self.detection.event_scores = np.delete(self.detection.event_scores, rows, axis=0)

                self.exclude_events = np.delete(self.exclude_events, rows, axis=0)
                self.use_for_avg = np.delete(self.use_for_avg, rows, axis=0)

        self.detection.singular_event_indices = np.where(self.use_for_avg == 1)[0]
        if not len(self.detection.singular_event_indices):
            self._warning_box(message='All events excluded for average. At least one has to remain, using all detected events instead!')

        if len(self.detection.event_locations) > 0:
            self.detection._eval_events()
            
            self.update_main_plot()

            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()
            self.num_events = self.detection.event_locations.shape[0]

        else:
            self.num_events = 0
            self._warning_box(message='All detected events were deleted.')


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
        pen = pg.mkPen(color=self.settings.colors[3], width=1)
        self.plotData = self.tracePlot.plot(self.trace.time_axis, self.trace.data, pen=pen, clear=True)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        label1 = 'Vmon' if self.recording_mode == 'current-clamp' else 'Imon'
        self.tracePlot.setLabel('left', label1, self.trace.y_unit)
        if self.was_analyzed and self.detection.event_locations.shape[0] > 0:
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[self.detection.event_peak_locations]
            pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
            self.plotDetected = self.tracePlot.plot(ev_positions, ev_peakvalues, pen=pen, symbol='o', symbolSize=8, 
                                                    symbolpen=self.settings.colors[0], symbolBrush=self.settings.colors[0])


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
        self.settings.convolve_win = int(settings_win.convolve_window.text())


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

            self.was_analyzed = False

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

            self.detection.detect_events(stride=self.settings.stride, eval=True, convolve_win=self.settings.convolve_win)

            self.was_analyzed = True
            pen = pg.mkPen(color=self.settings.colors[3], width=1)
            prediction_x = np.arange(0, len(self.detection.prediction)) * self.trace.sampling
            self.predictionPlot.plot(prediction_x, self.detection.prediction, pen=pen, clear=True)
            self.predictionPlot.plot([0, prediction_x[-1]], [self.settings.event_threshold, self.settings.event_threshold], 
                                     pen=pg.mkPen(color=self.settings.colors[0], style=Qt.DashLine, width=1))

        if self.detection.event_locations.shape[0] > 0:
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[self.detection.event_peak_locations]
            pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
            self.plotDetected = self.tracePlot.plot(ev_positions, ev_peakvalues, pen=pen, symbol='o', symbolSize=8, 
                                                    symbolpen=self.settings.colors[0], symbolBrush=self.settings.colors[0])

            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()

            # Set variables needed for event viewer to work.
            self.num_events = self.detection.event_locations.shape[0]
            self.exclude_events = np.zeros(self.num_events)
            self.use_for_avg = np.zeros(self.num_events, dtype=int)
            self.use_for_avg[self.detection.singular_event_indices] = 1

        else:
            self._warning_box(message='No events detected.')
            

    def show_event_viewer(self) -> None:
        """
        Start the event viewer.
        """
        if not hasattr(self, 'detection'):
            return
        if self.was_analyzed and self.num_events > 0:
            event_win = EventViewer(self)
            event_win.exec_()
        else:
            self._warning_box(message='Please load and analyze data first!')


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
        '''
        Plot events, histogram and average event.
        '''
        self.eventPlot.clear()
        self.eventPlot.setTitle('Detected events')
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        for event in self.detection.events:
            self.eventPlot.plot(time_data, event, pen=pg.mkPen(color=self.settings.colors[3], width=1))
        self.eventPlot.setLabel('bottom', 'Time', 's')
        self.eventPlot.setLabel('left', 'Amplitude', 'pA')

        y, x = np.histogram(self.detection.event_stats.amplitudes, bins='auto')
        curve = pg.PlotCurveItem(x, y, stepMode='center', fillLevel=0, brush=self.settings.colors[3])
        self.histogramPlot.clear()
        self.histogramPlot.setTitle('Amplitude histogram')
        self.histogramPlot.addItem(curve)
        self.histogramPlot.setLabel('bottom', 'Amplitude', 'pA')
        self.histogramPlot.setLabel('left', 'Count', '')

        ev_average = np.mean(self.detection.events[self.detection.singular_event_indices], axis=0)
        self.averagePlot.clear()
        self.averagePlot.setTitle('Average event waveform')
        time_data = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        self.averagePlot.plot(time_data, ev_average, pen=pg.mkPen(color=self.settings.colors[2], width=2))
        self.averagePlot.setLabel('bottom', 'Time', 's')
        self.averagePlot.setLabel('left', 'Amplitude', 'pA')

    
    def tabulate_results(self, tableWidget):
        tableWidget.clear()
        n_events = len(self.detection.event_stats.amplitudes)
        tableWidget.setHorizontalHeaderLabels(['Location', 'Amplitude', 'Area', 'Risetime', 'Decay'])
        tableWidget.setRowCount(n_events)
        for i in range(n_events):
            tableWidget.setItem(i, 0, QTableWidgetItem(f'{self.detection.event_locations[i] * self.detection.trace.sampling :.5f}'))
            tableWidget.setItem(i, 1, QTableWidgetItem(f'{self.detection.event_stats.amplitudes[i]:.5f}'))
            tableWidget.setItem(i, 2, QTableWidgetItem(f'{self.detection.event_stats.charges[i]:.5f}'))
            tableWidget.setItem(i, 3, QTableWidgetItem(f'{self.detection.event_stats.risetimes[i]:.5f}'))
            tableWidget.setItem(i, 4, QTableWidgetItem(f'{self.detection.event_stats.halfdecays[i]:.5f}'))
        tableWidget.show()


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

        self.paper = QLabel('Publication: <a href=\"https://doi.org/10.7554/eLife.98485.3\">miniML eLife paper 2024</a>')
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

        self.convolve_window = QLineEdit(str(parent.settings.convolve_win))
        self.convolve_window.setValidator(QIntValidator(1, 10000))

        self.layout = QFormLayout(self)
        self.layout.addRow('Stride length (samples)', self.stride)
        self.layout.addRow('Event length (samples)', self.ev_len)
        self.layout.addRow('Min. peak height (0-1)', self.thresh)
        self.layout.addRow('Model', self.model)
        self.layout.addRow('Event direction', self.direction)
        self.layout.addRow('Batch size', self.batchsize)
        self.layout.addRow('Filter window', self.convolve_window)

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


class EventViewer(QDialog):
    def __init__(self, parent=None):
        super(EventViewer, self).__init__(parent)

        self.resize(750, 600)

        self.detection = parent.detection
        self.settings = parent.settings
        self.num_events = parent.num_events
        self.exclude_events = parent.exclude_events
        self.use_for_avg = parent.use_for_avg

        self.layout = QGridLayout(self)
        self.layout.setColumnMinimumWidth(0, 200)
        self.layout.setColumnMinimumWidth(1, 200)
        self.layout.setColumnMinimumWidth(2, 225)
        self.layout.setRowMinimumHeight(1, 120)
        self.layout.setRowMinimumHeight(2, 160)
        self.layout.setRowMinimumHeight(3, 160)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)

        self.layout.addWidget(self.toolbar, 0, 0, 1, 3)

        self.testAction = QAction(QIcon('icons/arrow_back_24px_blue.svg'), 'Previous', self.toolbar)
        self.toolbar.addAction(self.testAction)
        self.testAction.triggered.connect(self.previous)
        self.deleteAction = QAction(QIcon('icons/clear_24px_blue.svg'), 'Delete event', self.toolbar)
        self.toolbar.addAction(self.deleteAction)
        self.deleteAction.triggered.connect(self.delete_event)
        self.excludeAction = QAction(QIcon('icons/hide_image_24px_blue.svg'), 'Exclude from average', self.toolbar)
        self.toolbar.addAction(self.excludeAction)
        self.excludeAction.triggered.connect(self.exclude_event)
        self.testAction = QAction(QIcon('icons/arrow_forward_24px_blue.svg'), 'Next', self.toolbar)
        self.toolbar.addAction(self.testAction)
        self.testAction.triggered.connect(self.next)

        self.tracePlot = pg.PlotWidget()
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Imon', '')
        self.layout.addWidget(self.tracePlot, 1, 0, 1, 3)

        self.eventPlot = pg.PlotWidget()
        self.eventPlot.showGrid(x=True, y=True, alpha=0.1)
        self.eventPlot.setLabel('bottom', 'Time', 's')
        self.eventPlot.setLabel('left', 'Imon', '')
        self.layout.addWidget(self.eventPlot, 2, 0, 2, 2)

        self.averagePlot = pg.PlotWidget()
        self.layout.addWidget(self.averagePlot, 2, 2, 1, 1)

        self.histPlot = pg.PlotWidget()
        self.layout.addWidget(self.histPlot, 3, 2, 1, 1)

        self.ind = 0
        self.left_buffer = int(self.detection.window_size / 2)
        self.right_buffer = int(self.detection.window_size * 1.5)
        self.filtered_data = self.detection.hann_filter(data=self.detection.trace.data, filter_size=self.detection.convolve_win)

        # create a downsampled trace for the event plot
        self.trace_x = np.arange(0, self.detection.trace.data.shape[0], 10) * self.detection.trace.sampling
        self.trace_y = self.detection.trace.data[::10]

        self.init_trace_plot()
        self.init_avg_plot()
        self.init_histogram_plot()
        self.update_event_plot()

        QBtn = (QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.close_event_viewer)
        self.buttonBox.rejected.connect(self.cancel_event_viewer)

        self.layout.addWidget(self.buttonBox, 4, 2, 1, 1)
        self.setWindowTitle('Event Viewer')
        self.setWindowModality(pg.QtCore.Qt.ApplicationModal)


    def cancel_event_viewer(self):
        self.exclude_events = 0
        self.use_for_avg = 1
        self.close()
    
    
    def close_event_viewer(self):
        rows = np.where(self.exclude_events == 1)[0]
        self.parent().delete_multiple_events(rows)
        self.close()


    def init_trace_plot(self):
        self.tracePlot.clear()
        trace = pg.PlotDataItem(self.trace_x, self.trace_y, pen=pg.mkPen(color=self.settings.colors[3], width=1))
        self.tracePlot.addItem(trace)
        self.tracePlot.setLabel('bottom', 'Time', 's')
        self.tracePlot.setLabel('left', 'Amplitude', self.detection.trace.y_unit)

        self.update_trace_plot()

    
    def init_avg_plot(self):
        self.averagePlot.clear()
        self.avg_time_ax = np.arange(0, self.detection.events[0].shape[0]) * self.detection.trace.sampling
        self.avg = pg.PlotDataItem(self.avg_time_ax, np.mean(self.detection.events[self.detection.singular_event_indices], axis=0), 
                              pen=pg.mkPen(color=self.settings.colors[2], width=2))
        self.averagePlot.addItem(self.avg)
        self.averagePlot.setLabel('bottom', 'Time', 's')
        self.averagePlot.setLabel('left', 'Amplitude', self.detection.trace.y_unit)


    def init_histogram_plot(self):
        self.histPlot.clear()
        y, x = np.histogram(self.detection.event_stats.amplitudes, bins='auto')
        self.curve = pg.PlotCurveItem(x, y, stepMode='center', fillLevel=0, brush=self.settings.colors[3])
        self.histPlot.addItem(self.curve)
        self.histPlot.setLabel('bottom', 'Amplitude', self.detection.trace.y_unit)
        self.histPlot.setLabel('left', 'Count', '')


    def update_avg_plot(self):
        self.avg.setData(self.avg_time_ax, np.mean(self.detection.events[self.use_for_avg == 1], axis=0))


    def update_histogram_plot(self):
        y, x = np.histogram(self.detection.event_stats.amplitudes[self.exclude_events == 0], bins='auto')
        self.curve.setData(x, y)


    def update_trace_plot(self):
        peak_loc = self.detection.event_peak_locations[self.ind]

        if hasattr(self, 'eventitem'):
            self.eventitem.setData([peak_loc * self.detection.trace.sampling, peak_loc * self.detection.trace.sampling],
                                   [np.min(self.detection.trace.data), np.max(self.detection.trace.data)])
        else:
            self.eventitem = pg.PlotDataItem([peak_loc * self.detection.trace.sampling, peak_loc * self.detection.trace.sampling],
                                             [np.min(self.detection.trace.data), np.max(self.detection.trace.data)],
                                             pen=pg.mkPen(color='orange', width=2, style=pg.QtCore.Qt.DotLine))
            self.tracePlot.addItem(self.eventitem)


    def update_event_plot(self):
        """
        Updates the event plot.
        """
        event_loc = self.detection.event_locations[self.ind]
        peak_loc = self.detection.event_peak_locations[self.ind]
        peak_val = self.detection.event_peak_values[self.ind]
        bsl = self.detection.event_bsls[self.ind]
        min_value_rise = self.detection.min_values_rise[self.ind]
        max_value_rise = self.detection.max_values_rise[self.ind]

        zero_point = event_loc - self.left_buffer
        sampling_ms = self.detection.trace.sampling * 1e3
        
        peaks_in_win = self.detection.event_peak_locations[
            np.logical_and(self.detection.event_peak_locations > peak_loc,
                           self.detection.event_peak_locations < event_loc + self.right_buffer)]
        
        rel_peak_loc = (peak_loc - zero_point) * sampling_ms
        rel_peak_loc_left = (peak_loc - self.detection.peak_spacer - zero_point) * sampling_ms
        rel_peak_loc_right = (peak_loc + self.detection.peak_spacer - zero_point) * sampling_ms
        rel_bsl_start = (self.detection.bsl_starts[self.ind] - zero_point) * sampling_ms
        rel_bsl_end = (self.detection.bsl_ends[self.ind] - zero_point) * sampling_ms
        rel_min_rise = (self.detection.min_positions_rise[self.ind] - (zero_point * self.detection.trace.sampling)) * 1e3
        rel_max_rise = (self.detection.max_positions_rise[self.ind] - (zero_point * self.detection.trace.sampling)) * 1e3
        
        if not np.isnan(self.detection.half_decay[self.ind]):
            decay_loc = int(self.detection.half_decay[self.ind])
            rel_decay_loc = (decay_loc - zero_point) * sampling_ms

        if len(peaks_in_win):
            rel_peaks_in_win = (peaks_in_win - zero_point) * sampling_ms

        data = self.detection.trace.data[zero_point:event_loc + self.right_buffer]
        filtered_data = self.filtered_data[zero_point:event_loc + self.right_buffer]
        time_ax = np.arange(0, data.shape[0]) * sampling_ms

        data_plot = self.eventPlot.plot(time_ax, data, pen=pg.mkPen(color='gray', width=2.5), clear=True)
        data_plot.setAlpha(0.5, False)
        if self.exclude_events[self.ind]:
            event_color = self.settings.colors[0]
        elif self.use_for_avg[self.ind] == 0:
            event_color = self.settings.colors[1]
        else:
            event_color = self.settings.colors[3]
        self.eventPlot.plot(time_ax, filtered_data, pen=pg.mkPen(color=event_color, width=2.5))

        if not self.exclude_events[self.ind]:
            bsl_times = [rel_bsl_start, rel_bsl_end]
            bsl_vals = [bsl, bsl]

            def plot_symbols(trace_plot, x, y, color, symbol, size):
                pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
                trace_plot.plot(x, y, pen=pen, symbol=symbol, symbolSize=size, symbolpen=color, symbolBrush=color)

            def plot_line(trace_plot, x, y, color, width, style):
                pen = pg.mkPen(color=color, width=width, style=style)
                trace_plot.plot(x, y, pen=pen)

            plot_symbols(self.eventPlot, bsl_times, bsl_vals, 'r', 'o', 10)
            plot_line(self.eventPlot, bsl_times, bsl_vals, 'r', 2.5, pg.QtCore.Qt.DotLine)
            plot_line(self.eventPlot, [rel_bsl_end, rel_peak_loc], bsl_vals, 'k', 2.5, pg.QtCore.Qt.DotLine)

            plot_symbols(self.eventPlot, [rel_min_rise, rel_max_rise], [min_value_rise, max_value_rise], 'magenta', 'o', 10)
            plot_line(self.eventPlot, [rel_min_rise, rel_max_rise], [min_value_rise, min_value_rise], 'magenta', 2.5, pg.QtCore.Qt.DotLine)
            plot_line(self.eventPlot, [rel_max_rise, rel_max_rise], [min_value_rise, max_value_rise], 'magenta', 2.5, pg.QtCore.Qt.DotLine)

            plot_symbols(self.eventPlot, [rel_peak_loc_left, rel_peak_loc_right, rel_peak_loc], [peak_val]*3, 'orange', ['x', 'x', 'o'], [12, 12, 10])
            if len(peaks_in_win):
                plot_symbols(self.eventPlot, rel_peaks_in_win, self.filtered_data[peaks_in_win], 'orange', 'o', 10)
            plot_line(self.eventPlot, [rel_peak_loc, rel_peak_loc], [peak_val, peak_val - self.detection.event_stats.amplitudes[self.ind]], 'orange', 2.5, pg.QtCore.Qt.DotLine)

            if not np.isnan(self.detection.half_decay[self.ind]):
                plot_symbols(self.eventPlot, [rel_decay_loc], [self.filtered_data[decay_loc]], 'green', 'o', 10)
                plot_line(self.eventPlot, [rel_peak_loc, rel_decay_loc], [self.filtered_data[decay_loc], self.filtered_data[decay_loc]], 'green', 2.5, pg.QtCore.Qt.DotLine)

        pen = pg.mkPen(color='k', width=1.5)

        color = 'green' if self.use_for_avg[self.ind] else 'red'
        text_str = f'event #{self.ind+1}/{self.num_events}: {"used for" if self.use_for_avg[self.ind] else "excluded from"} average'
        self.text = pg.TextItem(text_str, color=color, border=pen)
        self.eventPlot.addItem(self.text)
        self.text.setPos(0, np.max(data) + (np.max(data) - np.min(data))/10)

        self.eventPlot.setLabel('bottom', 'Time', 'ms')
        self.eventPlot.setLabel('left', 'Amplitude', self.detection.trace.y_unit)


    def previous(self):
        self.ind = (self.ind - 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()


    def delete_event(self):
        self.exclude_events[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.use_for_avg[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plot()


    def exclude_event(self):            
        self.use_for_avg[self.ind] = (self.use_for_avg[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plot()


    def next(self):
        self.ind = (self.ind + 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()


    def keyPressEvent(self, event):
        key = event.key()
        
        if key == Qt.Key_Right:  # forward key
            self.next()
        elif key == Qt.Key_Left:  # backward key
            self.previous()
        elif key == Qt.Key_M:  # 'm'
            self.delete_event()
        elif key == Qt.Key_N:  # 'n'
            self.exclude_event()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(__file__).parent.parent / 'minML_icon.png')))
    main = minimlGuiMain()
    extra = {'density_scale': '-1',}
    app.setStyleSheet(build_stylesheet(theme='light_blue.xml', invert_secondary=False, 
                                       extra=extra, template='miniml.css.template'))
    main.show()
    sys.exit(app.exec_())
