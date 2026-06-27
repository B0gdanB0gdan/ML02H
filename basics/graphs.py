from collections import deque


graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["B"],
    "E": ["B"],
    "F": ["C"],
    "G": ["C"],
}


def dfs(node: str):

    if node in visited:
        return
    
    visited.append(node)
    print(node)

    for neighbor in graph[node]:
        dfs(neighbor)


def bfs(queue):
    if not queue:
        return
   
    node = queue.popleft()

    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.append(neighbor)
            print(neighbor)
            queue.append(neighbor)

    bfs(queue)

print("DFS")
visited = []
dfs("A")
print("BFS")
visited = ["A"]
print("A")
queue = deque(["A"])
bfs(queue)