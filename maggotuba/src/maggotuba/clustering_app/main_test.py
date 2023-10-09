import numpy as np
from numpy.random import default_rng
from treewidget import ClusterTreeWidget

import sys
import PySide2
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QCheckBox, QRadioButton, QButtonGroup, QScrollArea, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter
from PySide2.QtGui import QPainter, QColor, QPen
from PySide2 import QtCore

from plotwidget import PointCloudDisplay

from pyqtgraph import PlotWidget

from tomatowrapper import TomatoLayout

from videogrid import VideoGrid

from app import App




if __name__ == '__main__':

    # create fake data
    rng = default_rng()
    centers = rng.normal(size=(100,1,2))
    radii = 0.5*rng.normal(size=(100,1,1))**2
    data = radii*rng.normal(size=(100,100,2)) + centers
    data = data.reshape(-1,2)

    ndata = len(data)
    xs = np.arange(5).reshape(1, 1, 1, -1)
    ys = np.zeros((ndata, 20, 1, 5))+0.2*rng.normal(size=(ndata,5)).reshape(-1,1,1,5)
    videodata = np.concatenate([np.broadcast_to(xs, ys.shape),ys], axis=2)
    videodata += 0.05*rng.normal(size=videodata.shape)

    labels = rng.choice(6, size=(len(data),1))

    app = App('title', 20, 20, data, data, labels, videodata, 20)

    app.start()