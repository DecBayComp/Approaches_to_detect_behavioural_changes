from PySide2.QtWidgets import QWidget, QSizePolicy
from PySide2.QtGui import QPainter, QPen
from PySide2 import QtCore

# Maybe at some point rewrite using QGraphicsScene ?
# Needs a rewrite anyway, this is real spaghetti code.

class ClusterTreeWidget(QWidget):
    activeNodeChanged = QtCore.Signal(tuple)

    def __init__(self, tmt, parent=None):
        super().__init__(parent)
        self.scale = 1.
        # self.setMinimumSize(400,400)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.colors = (QtCore.Qt.black,
                       QtCore.Qt.red,
                       QtCore.Qt.darkGreen,
                       QtCore.Qt.cyan,
                       QtCore.Qt.darkBlue,
                       QtCore.Qt.yellow)


        # intialize from tomato wrapper
        self.depth = tmt.max_clusters
        self.n_nodes = self.depth*(self.depth+1)//2
        self.hists = [tuple(l) for l in sum(tmt.histograms_in_graph, [])]

        # 1 indicates the parent node is up and to the right, -1 up and to the left
        self.parents = sum([[1+2*(tmt.parents_in_graph[i][j]-j) for j in range(i+1)] for i in range(self.depth)], []) # bad code.
        self.child_nodes = self.children_from_parents()
        self.active_nodes = [True for _ in range(self.n_nodes)]
        self.folded_nodes = [False for _ in range(self.n_nodes)]

 

        self.radius = min(self.width(), self.height())/3/self.depth
        self.heightDelta = 3*self.radius
        self.widthDelta = 3*self.radius

        self.startWidth = self.width()/2
        self.startHeight = 1.5*self.radius

        self.highlighted_node = None
        self.highlighted_node_t = None

        self.secondary_highlight = [False for _ in range(self.n_nodes)]

        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.currently_panning = False

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def index_node(self, i, j):
        return (i+1)*i//2 + j

    def children_from_parents(self):
        child_nodes = []
        for i in range(self.depth-1):
            for j in range(i+1):
                temp = []
                if self.parents[self.index_node(i+1,j)] == 1:
                    temp.append(self.index_node(i+1,j))
                if self.parents[self.index_node(i+1, j+1)] == -1:
                    temp.append(self.index_node(i+1,j+1))
                child_nodes.append(temp)

        return child_nodes + [[] for _ in range(self.depth)]


    # resizing hack ; probably not the right way to do it
    def event(self, event):
        if event.type() == QtCore.QEvent.Resize:
            self.resizeEventOverride(event)
        return super().event(event)

    # resizing hack ; probably not the right way to do it
    def resizeEventOverride(self, event):
        if event.oldSize().width() >= 0 and event.oldSize().height() >= 0:
            self.radius = min(self.width(), self.height())/3/self.depth
            self.heightDelta = 3*self.radius
            self.widthDelta = 3*self.radius

            self.startWidth = self.startWidth*self.width()/event.oldSize().width()
            self.startHeight = self.startHeight*self.height()/event.oldSize().height()

            self.repaint()

    def paintEvent(self, event):
        s = self.scale
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # draw the links
        
        startWidth = s*self.startWidth - s*self.widthDelta/2
        h = s*self.startHeight + s*self.heightDelta
        for i in range(1, self.depth):
            w = startWidth
            for j in range(i+1):
                if self.active_nodes[self.index_node(i,j)]:

                    parent = self.parents[self.index_node(i,j)]
                    parent_node = self.index_node(i-1, j-(parent==-1))
                    wp, hp = w + s*parent*self.widthDelta/2, h - s*self.heightDelta

                    if self.secondary_highlight[parent_node] and self.secondary_highlight[self.index_node(i,j)]:
                        pen = QPen(QtCore.Qt.red)
                        pen.setWidth(s*self.radius)
                        painter.setPen(pen)
                        painter.setBrush(QtCore.Qt.NoBrush)
                    else:
                        pen = QPen(QtCore.Qt.black)
                        pen.setWidth(s*self.radius/2)
                        painter.setPen(pen)
                        painter.setBrush(QtCore.Qt.NoBrush)

                    painter.drawLine(w, h, wp, hp)
                w += s*self.widthDelta
            startWidth -= s*self.widthDelta/2
            h += s*self.heightDelta


        # draw the pies
        startWidth = s*self.startWidth
        h = s*self.startHeight
        for i in range(self.depth):
            w = startWidth
            for j in range(i+1):
                if self.active_nodes[self.index_node(i,j)]:
                    if self.needs_drawing(i,j):
                        hist = self.hists[self.index_node(i,j)]
                        highlight = self.secondary_highlight[self.index_node(i, j)]
                        self.drawPie(painter, w, h, s*self.radius, hist, self.colors, highlight=highlight)
                w += s*self.widthDelta
            startWidth -= s*self.widthDelta/2
            h += s*self.heightDelta

        painter.end()

    def drawPie(self, painter, x, y, r, proportions, colors, highlight=False):
        assert len(colors) == len(proportions)

        if highlight:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtCore.Qt.red)
            painter.drawEllipse(x-1.3*r, y-1.3*r, 2.6*r, 2.6*r)

        total = sum(proportions)
        start = 0

        painter.setPen(QtCore.Qt.NoPen)
        for color, proportion in zip(colors, proportions):
            painter.setBrush(color)
            painter.drawPie(x-r, y-r, 2*r, 2*r, (start*16*360)/total, (proportion*16*360)/total)
            start += proportion
        pen = QPen(QtCore.Qt.black)
        pen.setWidth(r/10)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(x-r, y-r, 2*r, 2*r)



    def mousePressEvent(self, mouseEvent):
        node = self.getNodeFromCoordinates(mouseEvent.x(), mouseEvent.y())

        if node is not None:
            t, n = node
            
            if mouseEvent.button() == QtCore.Qt.RightButton:
                if self.folded_nodes[n]:
                    self.unfoldOne(n)
                else:
                    self.foldNode(n)

            elif mouseEvent.button() == QtCore.Qt.LeftButton:
                if self.highlighted_node != n:
                    self.activeNodeChanged.emit(t)
                    self.highlighted_node = n
                    self.highlighted_node_t = t
                    self.secondary_highlight_children()
        
        if mouseEvent.button() == QtCore.Qt.LeftButton:
            self.last_mouse_x = mouseEvent.x()
            self.last_mouse_y = mouseEvent.y()
            self.currently_panning = True        

        self.repaint()

        return super().mousePressEvent(mouseEvent)

    def mouseDoubleClickEvent(self, event):
        node = self.getNodeFromCoordinates(event.x(), event.y())
        if node is None:
            return super().mouseDoubleClickEvent(event)
        t, n = node

        self.unfoldAll(n)

        return super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            theta = event.angleDelta().y()
            f = (1+1e-3)**theta
            os = self.scale
            self.scale *= f
            # trick to zoom on mouse position
            self.startWidth = ((os*self.startWidth-event.x())*f + event.x())/self.scale
            self.startHeight = ((os*self.startHeight-event.y())*f + event.y())/self.scale
 
            self.repaint()

    def mouseMoveEvent(self, event):  
        s = self.scale
        if self.currently_panning:
            self.startHeight += (event.y() - self.last_mouse_y)/s
            self.startWidth  += (event.x() - self.last_mouse_x)/s
            self.repaint()

        self.last_mouse_x = event.x()
        self.last_mouse_y = event.y()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.currently_panning = False

    def resizeEvent(self, event):
        return super().resizeEvent(event)

    def getNodeFromCoordinates(self, x, y):
        s = self.scale
        i = round((y-s*self.startHeight)/(s*self.heightDelta))
        j = round((x-(s*self.startWidth-i*s*self.widthDelta//2))/(s*self.widthDelta))

        if i >= 0 and i < self.depth and j >= 0 and j <= i:
            if self.active_nodes[self.index_node(i,j)]:
                cy = s*self.startHeight + s*i*self.heightDelta
                cx = s*self.startWidth - s*i*self.widthDelta//2 + s*j*self.widthDelta
                if (x-cx)**2 + (y-cy)**2 <= (s*self.radius)**2:
                    if self.needs_drawing(i, j):
                        return (i,j), self.index_node(i,j)
        return None

    def foldNode(self, node):
        assert(self.active_nodes[node] and not(self.folded_nodes[node]))
        self.folded_nodes[node] = True
        toDeactivate = [c for c in self.child_nodes[node]]
        while toDeactivate:
            currentNode = toDeactivate.pop()
            self.active_nodes[currentNode] = False
            self.folded_nodes[currentNode] = True
            for c in self.child_nodes[currentNode]:
                toDeactivate.append(c)

    def unfoldOne(self, node):
        assert(self.folded_nodes[node] and self.active_nodes[node])
        while len(self.child_nodes[node]) == 1:
            self.active_nodes[node] = True
            self.folded_nodes[node] = False
            node = self.child_nodes[node][0]
        

        self.active_nodes[node] = True
        self.folded_nodes[node] = False
        
        for c in self.child_nodes[node]:
            self.active_nodes[c] = True

    def unfoldAll(self, node):
        assert(self.active_nodes[node])
        toActivate = [node]
        while toActivate:
            currentNode = toActivate.pop()
            self.active_nodes[currentNode] = True
            self.folded_nodes[currentNode] = False
            for c in self.child_nodes[currentNode]:
                toActivate.append(c)

    def secondary_highlight_children(self):
        node = self.highlighted_node
        self.secondary_highlight = [False for _ in range(self.n_nodes)]
        toHighlight = [node]
        while toHighlight:
            currentNode = toHighlight.pop()
            self.secondary_highlight[currentNode] = True
            for c in self.child_nodes[currentNode]:
                toHighlight.append(c)

    def needs_drawing(self, i, j):
        return (len(self.child_nodes[self.index_node(i, j)]) in [0,2] or
                        (i > 0 and len(self.child_nodes[self.index_node(i-1, j - (self.parents[self.index_node(i,j)]==-1))]) == 2))

    def setCoherentMerging(self, state):
        self.coherentMerging = bool(state)