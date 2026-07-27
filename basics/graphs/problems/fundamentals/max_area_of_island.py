"""
You are given an m x n binary matrix grid. 
An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) 
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
"""

def max_area_of_island(grid):
    """
    Time: O(m*n)
    Space: O(m*n)
    """
    m = len(grid)
    n = len(grid[0])
    directions = [(-1,0),(1,0),(0,1),(0,-1)]

    def dfs(grid, i, j, visited, cnt):
        visited.add((i, j))

        for dx, dy in directions:
            new_i, new_j = i+dx, j+dy
            if new_i < m and new_j < n and new_i >= 0 and new_j >= 0 and (new_i, new_j) not in visited:
                if grid[new_i][new_j] == 1:
                    cnt[0] += 1
                    dfs(grid, new_i, new_j, visited, cnt)

    max_area = 0
    visited = set()
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and (i, j) not in visited:
                cnt = [1]
                dfs(grid, i, j, visited, cnt) 
                max_area = max(max_area, cnt[0])
    return max_area


if __name__ == "__main__":
    grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,0,0,0],
            [0,1,1,0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,1,1,0,0,1,0,1,0,0],
            [0,1,0,0,1,1,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0,0,0]]
    print(max_area_of_island(grid))