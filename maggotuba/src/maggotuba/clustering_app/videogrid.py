import numpy as np
from numpy.random import default_rng
from scipy.fft import irfft

from PySide2.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QGridLayout
from PySide2 import QtCore


from pyqtgraph import PlotWidget, mkPen

import itertools


class VideoGrid(QWidget):
    def __init__(self, nrows, ncols, len_pres, len_pred, tmt, spineData, outlineData=None):
        super().__init__()
        rng = default_rng()
        self.nrows = nrows
        self.ncols = ncols
        self.spineData = spineData + 1e-4*(rng.random(size=spineData.shape)-.5) 
        # jitter to avoid unwanted plotting of an horizontal line when two successive points coincide

        self.outlineData = outlineData
        self.tmt = tmt

        self.plotters = [[PlotWidget() for _ in range(4)] for _ in range(2)]
        self.spines = [[None for _ in range(4)] for _ in range(2)]
        self.outlines = [[None for _ in range(4)] for _ in range(2)]
        self.midpoints = [[None for _ in range(4)] for _ in range(2)]
        self.heads = [[None for _ in range(4)] for _ in range(2)]
        self.time = 0
        self.spineToPlot = None
        self.outlineToPlot = None

        self.plotOutlines = False
        self.plotSpines = True
        self.plotPastFut = False

        self.len_pred = len_pred
        self.len_pres = len_pres


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_anim)

        vidlayout = QGridLayout()
        for i in range(2):
            for j in range(4):
                vidlayout.addWidget(self.plotters[i][j], i, j)
                self.plotters[i][j].setAspectLocked()
                self.plotters[i][j].enableAutoRange('xy', False)

        button_row = QHBoxLayout()
        self.button_labels = ['Save all videos',
                              'Save selected videos',
                              'Save displayed sample list', 
                              'Save selected sample list', 
                              'Save cluster list',
                              'Previous',
                              'Next']

        self.buttons = {s:QPushButton(s) for s in self.button_labels}
        self.buttons['Save all videos'].setEnabled(False)
        self.buttons['Save selected videos'].setEnabled(False)
        self.buttons['Save displayed sample list'].setEnabled(False)
        self.buttons['Save selected sample list'].setEnabled(False)
        self.buttons['Save cluster list'].setEnabled(False)
        self.buttons['Previous'].clicked.connect(self.decrementToPlotStart)
        self.buttons['Next'].clicked.connect(self.incrementToPlotStart)

        for button in self.buttons.values():
            button_row.addWidget(button)

        complete_layout = QVBoxLayout()
        complete_layout.addLayout(vidlayout)
        complete_layout.addLayout(button_row)

        self.setLayout(complete_layout)

        self.toPlotStart = 0
        self.toplot_indices = []
        self.some_data_is_loaded = False

    @QtCore.Slot()
    def activeNodeChanged(self, tup):
        self.some_data_is_loaded = True
        self.toPlotStart = 0
        self.tmt.n_clusters_ = tup[0]+1
        label = self.tmt.tmt_index[tup[0]][tup[1]]
        toPlot = np.argwhere(self.tmt.labels_ == label)
        self.toplot_indices = toPlot.flatten()

        self.spineToPlot = self.spineData[self.toplot_indices[self.toPlotStart:self.toPlotStart+8],:,:]
        self.outlineToPlot = self.build_outline_from_fourier()

        self.update()


    def update(self):
        rng = default_rng()
        self.time = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.plotters[i][j].clear()

        for (i,j), spinedata, outlinedata in zip(itertools.product(range(self.nrows), range(self.ncols)), self.spineToPlot, self.outlineToPlot):


            if self.plotOutlines:
                if not outlinedata[0]['x'].any():
                    continue

                self.outlines[i][j] = self.plotters[i][j].plot(outlinedata[0]['x'], outlinedata[0]['y'], pen=mkPen('w', width=5))
                self.plotters[i][j].setRange(xRange=(np.min([min(o['x']) for o in outlinedata])-.2, np.max([max(o['x']) for o in outlinedata])+.2),
                                         yRange=(np.min([min(o['y']) for o in outlinedata])-.2, np.max([max(o['y']) for o in outlinedata])+.2))
                if not(self.plotSpines):
                    self.spines[i][j] = self.plotters[i][j].plot(spinedata[0,:,0], spinedata[1,:,0], pen=None, symbolSize=5, symbolPen=None, symbolBrush=(255,0,0,255), brush=None)
                    self.heads[i][j] = self.plotters[i][j].plot(spinedata[0,-1:,0], spinedata[1,-1:,0], symbolSize=18, symbolPen=None, symbolBrush=(255,0,0,255))

            if self.plotSpines:
                self.spines[i][j] = self.plotters[i][j].plot(spinedata[0,:,0], spinedata[1,:,0], pen=mkPen('w', width=12), symbolSize=15, symbolPen=None, symbolBrush=(255,255,255,255))
                self.heads[i][j] = self.plotters[i][j].plot(spinedata[0,-1:,0], spinedata[1,-1:,0], symbolSize=18, symbolPen=None, symbolBrush=(255,0,0,255))
                self.plotters[i][j].setRange(xRange=(np.min(spinedata[0,:,:])-.2, np.max(spinedata[0,:,:])+.2),
                                             yRange=(np.min(spinedata[1,:,:])-.2, np.max(spinedata[1,:,:])+.2))

            self.midpoints[i][j] = self.plotters[i][j].plot([spinedata[0,2,0], spinedata[0,2,0]+1e-4*rng.random()], [spinedata[1,2,0], spinedata[1,2,0]+1e-4*rng.random()], pen=mkPen('b', width=3))

        self.timer.start(200)

    def update_anim(self):
        rng = default_rng()
        t = self.time
        for (i,j), spinedata, outlinedata in zip(itertools.product(range(self.nrows), range(self.ncols)), self.spineToPlot, self.outlineToPlot):
            if self.plotOutlines and self.outlines[i][j] is not None:
                self.outlines[i][j].setData(outlinedata[t]['x'], outlinedata[t]['y'])
                if not(self.plotSpines):
                    self.spines[i][j].setData(spinedata[0,:,t], spinedata[1,:,t])
            if self.plotSpines:
                self.spines[i][j].setData(spinedata[0,:,t], spinedata[1,:,t])
            
            self.heads[i][j].setData(spinedata[0,-1:,t], spinedata[1,-1:,t])

            if t > 1:
                self.midpoints[i][j].setData(spinedata[0,2,:t+1], spinedata[1,2,:t+1])
            else:
                self.midpoints[i][j].setData([spinedata[0,2,0], spinedata[0,2,0]+1e-4*rng.random()], [spinedata[1,2,0], spinedata[1,2,0]+1e-4*rng.random()])


        if self.plotPastFut:
            self.time = (self.time+1) % (2*self.len_pred+self.len_pres)
        else:
            self.time = ((self.time+1) % self.len_pres) + self.len_pred

    def decrementToPlotStart(self, foo):
        if not(self.some_data_is_loaded):
            return
        self.toPlotStart -= 8
        self.toPlotStart %= len(self.toplot_indices)
        if self.toPlotStart < 8:
            self.toPlotStart = 0

        self.spineToPlot = self.spineData[self.toplot_indices[self.toPlotStart:self.toPlotStart+8],:,:]
        self.outlineToPlot = self.build_outline_from_fourier()
        self.update()

    def incrementToPlotStart(self, foo):
        if not(self.some_data_is_loaded):
            return
        self.toPlotStart += 8
        self.toPlotStart %= len(self.toplot_indices)
        if self.toPlotStart < 8:
            self.toPlotStart = 0

        self.spineToPlot = self.spineData[self.toplot_indices[self.toPlotStart:self.toPlotStart+8],:,:]
        self.outlineToPlot = self.build_outline_from_fourier()
        self.update()

    def build_outline_from_fourier(self):
        outlineToPlot = []
        specsToPlot = [self.outlineData[i] for i in self.toplot_indices[self.toPlotStart:self.toPlotStart+8]]
        for specs in specsToPlot:
            outline_list = []
            for t in range(len(specs['fourier_x'])):
                outline = {}
                if np.isnan(specs['fourier_x']).any():
                    outline['x'] = np.array([])
                    outline['y'] = np.array([])
                else:
                    outline['x'] = irfft(128*specs['fourier_x'][t,:], 128)
                    outline['y'] = irfft(128*specs['fourier_y'][t,:], 128)
                outline_list.append(outline)
            outlineToPlot.append(outline_list)
        return outlineToPlot


