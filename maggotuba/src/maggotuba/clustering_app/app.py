import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QCheckBox, QHBoxLayout, QVBoxLayout, QSplitter
from PySide2 import QtCore

from maggotuba.clustering_app.treewidget import ClusterTreeWidget
from maggotuba.clustering_app.plotwidget import PointCloudDisplay
from maggotuba.clustering_app.tomatowrapper import TomatoLayout
from maggotuba.clustering_app.videogrid import VideoGrid


class App(QApplication):
    def __init__(self, title, len_pres, len_pred, clusteringData, plotData, labels, spineData, depth, outlineData=None, plotOutlines=False, plotSpines=True, plotPastFut=False):
        super().__init__([title])
    
        tmt = TomatoLayout(clusteringData, labels, max_n_clusters=50 if depth is None else depth)

        topwidget = QWidget()
        tree = ClusterTreeWidget(tmt)
        plotWidget = PointCloudDisplay(plotData, tmt)

        tree.activeNodeChanged.connect(plotWidget.activeNodeChanged)
        coherentMergingButton = QCheckBox('Enforce coherent cluster merging')
        coherentMergingButton.setEnabled(False)
        coherentMergingButton.stateChanged.connect(tree.setCoherentMerging)

        tree_and_buttons_layout = QVBoxLayout()
        tree_and_buttons_layout.addWidget(coherentMergingButton)
        tree_and_buttons_layout.addWidget(tree)

        plotlayout = QHBoxLayout()
        plotlayout.addLayout(tree_and_buttons_layout)
        plotlayout.addWidget(plotWidget)
        topwidget.setLayout(plotlayout)

        videos = VideoGrid(2,4, len_pres, len_pred, tmt, spineData, outlineData)
        videos.plotOutlines = plotOutlines
        videos.plotSpines = plotSpines
        videos.plotPastFut = plotPastFut

        tree.activeNodeChanged.connect(videos.activeNodeChanged)

        splitter = QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)

        splitter.addWidget(topwidget)
        splitter.addWidget(videos)

        splitterHeight = splitter.height()
        splitter.setSizes([0.65*splitterHeight, 0.35*splitterHeight])



        main_window = QMainWindow()
        main_window.setCentralWidget(splitter)

        self.main_window = main_window

    def start(self):
        self.main_window.show()
        sys.exit(self.exec_())    