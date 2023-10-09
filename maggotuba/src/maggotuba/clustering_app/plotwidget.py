from PySide2 import QtCore
from pyqtgraph import PlotWidget
import numpy as np

class PointCloudDisplay(PlotWidget):
    def __init__(self, data, tmt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.tmt = tmt
        self.highlighted = np.full((len(data)), True)
        self.plot(self.data[:,0], self.data[:,1], pen=None, symbolSize=3, symbolPen=None, symbolBrush=(255,255,255,100))

    @QtCore.Slot()
    def activeNodeChanged(self, tup):
        self.tmt.n_clusters_ = tup[0]+1
        self.highlighted[:] = False
        label = self.tmt.tmt_index[tup[0]][tup[1]]
        self.highlighted[self.tmt.labels_ == label] = True
        return self.update()

    def update(self):
        self.clear()
        self.plot(self.data[np.logical_not(self.highlighted),0], self.data[np.logical_not(self.highlighted),1], pen=None, symbolSize=3, symbolPen=None, symbolBrush=(255,255,255,100))
        self.plot(self.data[self.highlighted,0], self.data[self.highlighted,1], pen=None, symbolSize=3, symbolPen=None, symbolBrush=(255,  0,  0,100))

        return super().update()