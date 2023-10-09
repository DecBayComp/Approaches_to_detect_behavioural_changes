from gudhi.clustering.tomato import Tomato
import numpy as np

class TomatoLayout(Tomato):
    def __init__(self, data, labels, max_n_clusters=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.fit(data)
        max_n_clusters = min(50, max_n_clusters)
        self.max_clusters = max_n_clusters

        ###########################
        #
        #    Everything below is horribly hacky and sub optimal, but works. Good enough.
        #
        #################

        # determine parents and splitting nodes
        parent_cluster = []
        previous_labels = [-1]
        splitting_nodes = []
        for n_clusters in range(1, max_n_clusters+1):
            parent_cluster.append([])
            self.n_clusters_ = n_clusters
            tmt_labels = self.labels_
            for cluster in range(n_clusters):
                parent_cluster[-1].append(int(previous_labels[np.argwhere(tmt_labels==cluster)[0,0]]))
            previous_labels = tmt_labels.copy()
            values, counts = np.unique(parent_cluster[-1], return_counts=True)
            splitting_node = values[counts==2]
            splitting_nodes.append(splitting_node)
        splitting_nodes = [s.item() for s in splitting_nodes[1:]]

        # determine graph index
        graph_index = [[0]]
        for n_cl, splitting_node, parent_nodes in zip(range(len(splitting_nodes)), splitting_nodes, parent_cluster[1:]):
            graph_index.append([])

            splitting_node_visited = 0
            for parent_node in parent_nodes:
                if graph_index[n_cl][parent_node] < graph_index[n_cl][splitting_node]:
                    graph_index[-1].append(graph_index[n_cl][parent_node])
                elif graph_index[n_cl][parent_node] > graph_index[n_cl][splitting_node]:
                    graph_index[-1].append(graph_index[n_cl][parent_node]+1)
                else:
                    graph_index[-1].append(graph_index[n_cl][parent_node]+splitting_node_visited)
                    splitting_node_visited += 1

        # Compute inverse mapping
        tmt_index = []
        for n_cl in range(max_n_clusters):
            tmt_index.append((n_cl+1)*[0])
            for i, j in enumerate(graph_index[n_cl]):
                tmt_index[-1][j] = i

        self.graph_index = graph_index
        self.tmt_index = tmt_index
        self.parents_in_tmt = parent_cluster

        # Compute histograms and totals for succesive merge levels
        histograms = []
        totals = []
        for n_clusters in range(1, max_n_clusters+1):
            histograms.append([])
            totals.append([])
            self.n_clusters_ = n_clusters
            tmt_labels = self.labels_
            for cluster in range(n_clusters):
                unique, counts = np.unique(labels[tmt_labels==cluster], return_counts=True)
                histo = np.zeros(6, dtype=int)
                histo[unique.astype(int)] = counts
                histograms[-1].append(histo)
                totals[-1].append(np.sum(tmt_labels==n_clusters))
        total = totals[0][0]

        self.histograms_in_tmt = histograms
        self.totals_in_tmt = totals

        # convert tmts to graphs
        self.parents_in_graph = [[self.parents_in_tmt[i][self.tmt_index[i][j]] for j in range(i+1)] for i in range(len(self.parents_in_tmt))]
        self.parents_in_graph = [[self.graph_index[i-1][j] if i > 0 else 0 for j in self.parents_in_graph[i]] for i in range(len(self.parents_in_tmt))]
        self.histograms_in_graph = [[self.histograms_in_tmt[i][self.tmt_index[i][j]] for j in range(i+1)] for i in range(len(self.histograms_in_tmt))]
        self.totals_in_graph = [[self.totals_in_tmt[i][self.tmt_index[i][j]] for j in range(i+1)] for i in range(len(self.totals_in_tmt))]