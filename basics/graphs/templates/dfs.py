from collections import deque


def dfs(graph, node, visited):

    if node in visited:
        return

    visited.add(node)

    print(node)

    for neighbor in graph[node]:
        dfs(graph, neighbor, visited)


def dfs_iter(graph, start):

    stack = [start] # LIFO
    visited = set()

    while stack:
        node = stack.pop()

        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
                visited.add(node)


if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A'],
        'D': ['B']
    }
    dfs(graph, 'A', set())
    print("\n")
    dfs_iter(graph, 'A')

