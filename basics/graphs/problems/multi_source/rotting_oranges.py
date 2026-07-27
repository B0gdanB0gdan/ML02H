"""
You are given an m x n grid where each cell can have one of three values:

0 representing an empty cell,
1 representing a fresh orange, or
2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. 
If this is impossible, return -1.
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
"""

from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:

    m = len(grid)
    n = len(grid[0])

    queue = deque()
    fresh_count = 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j, 0))
            elif grid[i][j] == 1:
                fresh_count += 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    max_minutes = 0
    while queue:
        i, j, minute  = queue.popleft()
        
        visited.add((i, j))
        max_minutes = max(max_minutes, minute)

        for dx, dy in directions:
            new_i, new_j = i + dx, j + dy
            if 0 <= new_i < m and 0 <= new_j < n and (new_i, new_j) not in visited:
                if grid[new_i][new_j] == 1:
                    grid[new_i][new_j] = 2
                    fresh_count -= 1
                    queue.append((new_i, new_j, m+1))

    return max_minutes if fresh_count == 0 else -1


if __name__ == "__main__":
    print(oranges_rotting(
        grid =[
            [2,1,1],
            [1,1,0],
            [0,1,1]
        ]
    ))