"""
You have a set of nodes and a bunch of weighted edges connecting some of them.
You want to connect all the nodes together (so every node is reachable from every other), 
using the minimum possible total edge weight - and using as few edges as possible to do it.

Both Kruskal's and Prim's assume an undirected, weighted, connected graph

Kruskal: Sort all edges globally by weight, then greedily add the cheapest edge 
that doesn't create a cycle with Union-Find without caring where in the graph each edge is.

Prim: Start from one node and grow a single connected tree outward, 
at each step adding the cheapest edge that connects the current tree to any node not yet in it (found via a min-heap) expanding current tree rather than merging separate pieces.
"""

from .union_find import UnionFind
import heapq


def kruskal_mst(edges, n):
    """
    Time: O(ElogE for sorting + (E*O(1))) = O(ElogE)
    Space: O(E for edge list and V for union-find arrays) = O(V+E)
    """
    edges.sort(key=lambda e: e[2])
    uf = UnionFind(n)

    total_weight = 0
    mst_edges = []
    for u, v, weight in edges:
        if uf.union(u, v): 
            total_weight += weight
            mst_edges.append((u, v, weight))
            if len(mst_edges) == n-1:
                break

    return total_weight, mst_edges


def prim_mst(graph, start):
    visited = set()
    heap = [(0, start, -1)] # -1 means no parent
    total_weight = 0
    mst_edges = []
    while heap:

        w, node, parent = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if parent != -1:
            mst_edges.append((parent, node))

        total_weight += w

        for neighbor, w in graph[node]:
            if neighbor not in visited:
                heapq.heappush(heap, (w, neighbor, node))

    return mst_edges, total_weight


if __name__ == "__main__":
    edges = [(0, 1, 1), (1, 2, 3), (0, 3, 2), (3, 2, 1), (2, 4, 4)]
    print(kruskal_mst(edges, n=5))

    graph = {
        0: [(1, 1), (3, 2)],
        1: [(0, 1), (2, 3)],
        2: [(1, 3), (3, 1), (4, 4)],
        3: [(0, 2), (2, 1)],
        4: [(2, 4)],
    }
    print(prim_mst(graph, start=0))