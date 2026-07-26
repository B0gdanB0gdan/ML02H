"""
For DAG: O(V+E)
Avoid even Dijkstra: O(E log V)
"""

from collections import deque


def shortest_path_dag(graph, n, start):
    # Step 1: topological sort (Kahn's algorithm)
    topo_order = []
    in_degree = [0] * n
    for node in range(n):
        for (neighbor, d) in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in range(n) if in_degree[node] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor, d in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        

    # Step 2: process nodes in topological order, relaxing edges
    dist = [float('inf')] * n
    dist[start] = 0

    for node in topo_order:
        if dist[node] == float('inf'):
            continue   # unreachable from start, skip
        for neighbor, weight in graph[node]:
            # in a path from start to node v
            dist[neighbor] = min(dist[neighbor], dist[node] + weight)

    return dist


if __name__ == "__main__":
    graph = {
        0: [(1, 1)],   # X -> Y
        1: [(2, 1)],   # Y -> A
        2: [(3, 1)],   # A -> B
        3: [(4, 1)],   # B -> C
        4: [(5, 1)],   # C -> D
        5: [],
    }
    n = 6
    start = 2  # A
    print("dist:", shortest_path_dag(graph, n, start))
    print("expected: X,Y = inf, A=0, B=1, C=2, D=3")
