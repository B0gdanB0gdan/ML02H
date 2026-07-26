"""
Weakly + Strongly connected
"""

def con_components_und(graph):
    """
    for connected components in an undirected graph
    """
    n = len(graph)
    visited = set()
    count = 0

    def dfs(node, comp):
        visited.add(node)
        comp.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, comp)

    comps = []
    for node in range(n):
        if node not in visited:
            count += 1
            comp = []
            dfs(node, comp)
            comps.append(comp)

    return count, comps


def con_components_weak(graph):
    graph_weak = {}
    for k, l in graph.items():
        graph_weak[k] = l
        for i in l:
            if i not in graph_weak:
                graph_weak[i] = [k]
            else:
                graph_weak[i].append(k)

    return con_components_und(graph_weak)


def con_components_strong(graph):
    """Kosaraju's algo"""
    pass


if __name__ == "__main__":
    und_graph = {
        0: [1, 2],
        1: [0],
        2: [0, 3],
        3: [2],
        4: [5],
        5: [4]
    }

    print(con_components_und(und_graph))
    dir_graph = {
        0: [1, 3],
        1: [2],
        2: [],
        3: [2],
        4: [5],
        5: []
    }
    print(con_components_weak(dir_graph))
    


