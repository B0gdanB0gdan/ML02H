"""
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
write a function to check whether these edges make up a valid tree.

Example:
Input:
n = 5
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]

Output:
true
"""

from collections import defaultdict


def valid_tree(n: int, edges: list[list[int]]) -> bool:

    if len(edges) != n-1:
        return False

    # build adjacency list for traversal
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def dfs(graph, node, parent, visited):
        
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(graph, neighbor, node, visited):
                    return True
            elif neighbor != parent:
                return False
        return False

    has_cycle = False
    visited = set()
    for node in range(n):
        if node not in visited:
            has_cycle = has_cycle or dfs(graph, node, -1, visited)
    return not has_cycle



if __name__ == "__main__":
    edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
    print(valid_tree(5, edges))
    edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
    print(valid_tree(5, edges))