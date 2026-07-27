"""
There are n cities numbered from 0 to n-1. 
Given the array edges where edges[i] = [fromi, toi, weighti] represents a bidirectional and weighted edge 
between cities fromi and toi, and given the integer distanceThreshold.

Return the city with the smallest number of cities that are reachable through some path and whose distance is at most distanceThreshold, 
If there are multiple such cities, return the city with the greatest number.

Notice that the distance of a path connecting cities i and j is equal to the sum of the edges' weights along that path.

Input: n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4
Output: 3
"""

def find_the_city(n: int, edges: list[list[int]], distance_threshold: int):

    dist = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    for u, v, weight in edges:
        dist[u][v] = dist[v][u] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    result = {i: 0 for i in range(n)}
    for i in range(n):
        for j in range(n):
            if j != i and dist[i][j] <= distance_threshold:
                result[i] = result.get(i, 0) + 1

    return min(result, key=lambda k: (result[k], -k))


if __name__ == "__main__":
    print(find_the_city(
        n=4,
        edges=[[0,1,3],[1,2,1],[1,3,4],[2,3,1]],
        distance_threshold=4
    ))