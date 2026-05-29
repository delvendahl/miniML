"""
Heka Patchmaster .dat file browser 
Adapted from https://github.com/campagnola/heka_reader

"""

import os, sys
import pyqtgraph as pg
import numpy as np
import HekaReader

app = 0
app = pg.mkQApp()

# Configure Qt GUI:

# Main window + splitters to let user resize panes
win = pg.QtWidgets.QWidget()
layout = pg.QtWidgets.QGridLayout()
win.setLayout(layout)
hsplit = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Horizontal)
layout.addWidget(hsplit, 0, 0)
vsplit = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Vertical)
hsplit.addWidget(vsplit)
w1 = pg.QtWidgets.QWidget()
w1l = pg.QtWidgets.QGridLayout()
w1.setLayout(w1l)
vsplit.addWidget(w1)

# Button for loading .dat file
load_btn = pg.QtWidgets.QPushButton("Load...")
w1l.addWidget(load_btn, 0, 0)

# Tree for displaying .pul structure
tree = pg.QtWidgets.QTreeWidget()
tree.setHeaderLabels(['Node', 'Label'])
tree.setColumnWidth(0, 200)
w1l.addWidget(tree, 1, 0)

# Tree for displaying metadata for selected node
data_tree = pg.DataTreeWidget()
vsplit.addWidget(data_tree)

# Plot for displaying trace data
plot = pg.PlotWidget()
hsplit.addWidget(plot)

# Resize and show window
hsplit.setStretchFactor(0, 400)
hsplit.setStretchFactor(1, 600)
win.resize(1200, 800)
win.show()


def load_clicked():
    # Display a file dialog to select a .dat file
    file_name = pg.QtWidgets.QFileDialog.getOpenFileName()
    if file_name == '':
        return
    load(file_name[0])

load_btn.clicked.connect(load_clicked)


def load(file_name):
    """Load a new .dat file into the browser.
    """
    global bundle, tree_items

    # Read the bundle header
    # (no data is read at this time)
    bundle = HekaReader.Bundle(file_name)

    # Clear the tree and update to show the structure provided in the embedded
    # .pul file
    tree.clear()
    update_tree(tree.invisibleRootItem(), [])
    replot()


def update_tree(root_item, index):
    """Recursively read tree information from the bundle's embedded .pul file
    and add items into the GUI tree to allow browsing.
    """
    global bundle
    root = bundle.pul
    node = root
    for i in index:
        node = node[i]
    node_type = node.__class__.__name__
    if node_type.endswith('Record'):
        node_type = node_type[:-6]
    try:
        node_type += str(getattr(node, node_type + 'Count'))
    except AttributeError:
        pass
    try:
        node_label = node.Label
    except AttributeError:
        node_label = ''
    item = pg.QtWidgets.QTreeWidgetItem([node_type, node_label])
    root_item.addChild(item)
    item.node = node
    item.index = index
    if len(index) < 2:
        item.setExpanded(True)
    for i in range(len(node.children)):
        update_tree(item, index + [i])


def replot():
    """Show data associated with the selected tree node.

    For all nodes, the meta-data is updated in the bottom tree.
    For trace nodes, the data is plotted.
    """
    plot.clear()
    data_tree.clear()

    selected = tree.selectedItems()
    if len(selected) < 1:
        return

    # update data tree
    sel = selected[0]
    fields = sel.node.get_fields()
    data_tree.setData(fields)

    # plot all selected
    for sel in selected:
        index = sel.index
        if len(index) < 4:
            return

        trace = sel.node
        plot.setLabels(bottom=('Time', trace.XUnit), left=(trace.Label, trace.YUnit))
        data = bundle.data[index]
        time = np.linspace(trace.XStart, trace.XStart + trace.XInterval * (len(data)-1), len(data))
        plot.plot(time, data)


# replot when ever the user selects a new item
tree.itemSelectionChanged.connect(replot)

# load Heka's demo bundle if it is present
demo = 'B_2020-12-04_011.dat'
if os.path.isfile(demo):
    load(demo)

# Start the Qt event loop unless the user is in an interactive prompt
if sys.flags.interactive == 0:
    app.exec_()
