import numpy as np
import torch
import networkx as nx
import random

class Graph:
    def __init__(self, gp_idx=0):
        if gp_idx==0: # 7*7 grid, T=7
            #random.seed(4)
            rdm=random.Random(4)
            m = 7
            n = 7
            g = nx.grid_2d_graph(m, n)
            g = nx.convert_node_labels_to_integers(g, first_label=1)

            map_adjlist = nx.to_dict_of_lists(g)
            intra_nodes = [i for i in g.nodes() if len(map_adjlist[i]) == 4]

            p = 0.5
            to_remove = []
            for e in g.edges():
                if rdm.random() >= p:
                    to_remove.append(e)
            g.remove_edges_from(to_remove)

            q = 0.1

            def other_nodes(node, n=n):
                return node-n-1, node-n+1, node+n-1, node+n+1
            add_edges = []
            for node in intra_nodes:
                for other_node in other_nodes(node):
                    if rdm.random() <= q:
                        add_edges.append((node, other_node))
            g.add_edges_from(add_edges)

            g.add_edge(20, 21)
            map_adjlist = nx.to_dict_of_lists(g)
            max_actions = 0
            for node in map_adjlist:
                map_adjlist[node].append(node)
                map_adjlist[node].sort()
                if len(map_adjlist[node]) > max_actions:
                    max_actions = len(map_adjlist[node])
            self.num_nodes = len(map_adjlist)
            self.adjlist = map_adjlist
            self.defender_init = [(11, 23, 27, 39)]
            self.attacker_init = [25]
            self.exits = [4, 36, 45, 48, 7, 28, 8, 29, 1, 34]
            self.num_defender = len(self.defender_init[0])
            self.max_actions = pow(max_actions, self.num_defender)
            self.degree=max_actions
            self.graph = g
            self.size = [m, n]
            self.time_horizon = 7
        else:
            raise NotImplementedError()
