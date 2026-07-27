"""
Imagine tasks with dependencies: "you must finish task A before starting task B." 
A topological sort produces a valid linear ordering of all tasks such that every dependency
is respected i.e. for every edge u -> v (meaning "u must come before v"), 
u appears before v in the final ordering.

This only makes sense on a DAG (Directed Acyclic Graph)

Kahn's algorithm:
The core idea: 
a node is "ready to be placed in the order" once all of its prerequisites have already been placed. 
Track how many unmet prerequisites (in-degree) each node has; 
repeatedly pull out nodes with zero remaining prerequisites, and each time you place one, 
decrement the in-degree of everything it points to (since one of their prerequisites is now satisfied).

BFS based
Time: O(V+E)
Space: O(V+E)


An edge in Kahn's algo is going from a node that comes FIRST to the next that comes SECOND.
"""
from collections import deque


def topo_sort_kahn(graph, n):
    in_degree = [0] * n

    # upade number of incoming edges
    for node in range(n):
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in range(n) if in_degree[node] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node) 
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1 # everything that depends on it decreases by 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else None