"""
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
The distance between two cells sharing a common edge is 1.
Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
Output: [[0,0,0],[0,1,0],[1,2,1]]

"""

from collections import deque


def dist_nearest_zero(mat):

    queue = deque()
    visited = set()

    m = len(mat)
    n = len(mat[0])
    distances = [[0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0: # once we hit 1 with a wave nearest is found
                queue.append((i, j))
                visited.add((i, j))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    wave = 0
    while queue:
        for _ in range(len(queue)):
            i, j = queue.popleft()
            if mat[i][j] == 1:
                distances[i][j] = wave
             
            for dx, dy in directions:
                new_i, new_j = i + dx, j + dy
                if 0 <= new_i < m and 0 <= new_j < n and (new_i, new_j) not in visited:
                    visited.add((new_i, new_j))
                    queue.append((new_i, new_j)) 
        wave += 1
    return distances


if __name__ == "__main__":
    mat = [[0,0,0],[0,1,0],[1,1,1]]
    print(dist_nearest_zero(mat))