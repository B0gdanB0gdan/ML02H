"""
Ways of representing a graph:

1. Edge list:
edges = [(S1,T1), (S2, T2), ...]
or
edges_weighted = [(S1, T1, W1), (S2, T2, W2), ...]

Best for: Kruskal's MST algorithm (needs to sort edges by weight globally, doesn't care about per-node adjacency), 
and Bellman-Ford (relaxes all edges each round, doesn't need per-node lookup either). 

Rarely useful for BFS/DFS since you'd have to scan the whole list to find a node's neighbors.

2. Adjacency Matrix
matrix = [[0] * n for _ in range(n)]
for u, v, weight in edges:
    matrix[u][v] = weight
    matrix[v][u] = weight # omit if directed

Space: O(V*V) - wasteful for sparse graphs but gives O(1) edge existence checks (if matrix[i][j]:).
3. Adjacency List

from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

Or if nodes are 0 ... n-1 integers
graph = [[] for _ in range(n)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

Space: O(V + E)
"""

class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def edge_list_to_adjacency_list(graph, directed=False):
    pass

def adjacency_list_to_adjacency_matrix(graph):
    pass

def adjacency_matrix_to_list(graph):
    pass

def adjacency_list_to_edge_list(graph):
    pass