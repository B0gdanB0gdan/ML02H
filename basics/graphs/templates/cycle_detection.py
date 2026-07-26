def has_cycle_undirected(graph):

    def dfs(graph, node, prev, visited):

        visited.add(node)

        for neighbor in graph[node]:

            if neighbor not in visited:
                if dfs(graph, neighbor, node, visited):
                    return True

            elif neighbor != prev:
                return True

        return False

    n = len(graph)
    visited = set()
    has_cycle = False
    for node in range(n):
        if node not in visited:
            has_cycle = has_cycle or dfs(graph, node, -1, visited)
    return has_cycle


def has_cycle_directed(graph):
    # not visited, visiting, visited
    WHITE, GRAY, BLACK = 0, 1, 2

    def dfs(graph, node, color):

        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == WHITE:
                if dfs(graph, neighbor, color):
                    return True
            elif color[neighbor] == GRAY:
                return True

        color[node] = BLACK
        return False

    n = len(graph)
    has_cycle = False
    color = [WHITE] * n
    for node in range(n):
        if color[node] == WHITE:
            has_cycle = has_cycle or dfs(graph, node, color)

    return has_cycle


if __name__ == "__main__":
    graph = {
        0: [1],
        1: [0, 2],
        2: [1, 3, 4],
        3: [2],
        4: [2]
    }
    print(has_cycle_undirected(graph))
    graph = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3, 4],
        3: [2],
        4: [0, 2]
    }
    print(has_cycle_undirected(graph))

    graph = {
        0: [1],
        1: [2],
        2: [3],
        3: [0]
    }
    print(has_cycle_directed(graph))
    graph = {
        0: [1],
        1: [2],
        2: [3],
        3: []
    }
    print(has_cycle_directed(graph))