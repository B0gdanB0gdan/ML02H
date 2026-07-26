"""
dist from start to all nodes
allows negative edges but no negative cycles
Time: O(VE)
Space: O(V)

Bellman-Ford's solution: don't be clever, just repeat the relaxation process enough times 
that the correct answer is guaranteed to propagate everywhere.
Specifically: any shortest path between two nodes uses at most V-1 edges.
So if you relax every edge, V-1 times, in the worst case, the shortest distances are guaranteed to have fully propagated through the graph by the end.

A shortest path never repeats a vertex
A simple path has at most V-1 edges
"""

def bellman_ford(n, edges, start):
    # edges = list of (u, v, weight)

    dist = [float('inf')] * n 
    dist[start] = 0 # Everything else is infinity.

    for _ in range(n - 1):
        for u, v, weight in edges:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight

    # another relaxation is still possible only with negative cycles
    for u, v, weight in edges:
        if dist[u] + weight < dist[v]:
            return None

    return dist

    