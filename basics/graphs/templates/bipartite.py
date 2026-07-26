"""
Color nodes of each edge with different colors
"""


def is_bipartite(graph):

    def dfs(graph, node, color):

        for neighbor in graph[node]:

            if color[neighbor] == -1:
                color[neighbor] = 1 - color[node]
                if not dfs(graph, neighbor, color):
                    return False
            elif color[neighbor] == color[node]:
                return False

        return True  

    n = len(graph)
    is_bip = True
    color = [-1] * n
    for node in range(n):
        if color[node] == -1:
            color[node] = 0
            is_bip = is_bip and dfs(graph, node, color)    
    return is_bip


if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0],
        2: [0, 3, 5],
        3: [2],
        4: [5],
        5: [4, 2]
    }

    print(is_bipartite(graph))

    graph = {
            0: [1],
            1: [2],
            2: [0]
        }

    print(is_bipartite(graph))