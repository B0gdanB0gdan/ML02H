"""
dist between all pairs
allows negative edges but no negative cycles
Time: O(V^3)
Space: O(V^2)
"""

def floyd_warshall(n, edges):

    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, weight in edges:
        dist[u][v] = weight

    for k in range(n):
        # first consider paths that use only vertex 0 as an intermediate, then vertices 0, 1 then 0,1,2 and so on
        for i in range(n):
            for j in range(n):
                # dist[i][k] = shortest path from i to k using only intermediate vertices {0,1,...,k-1}
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist