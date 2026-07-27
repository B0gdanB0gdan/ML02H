"""
Given an m x n 2D binary grid grid which represents a map of 
'1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting 
adjacent lands horizontally or vertically. 
You may assume all four edges of the grid are all surrounded by water.

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
"""


def number_of_islands(grid):
    """
    Time: O(m*n)
    Space: O(m*n)
    """
    m = len(grid)
    n = len(grid[0])
    directions = [(-1,0),(1,0),(0,1),(0,-1)]


    def dfs(grid, i, j, visited):
        visited.add((i, j))

        for dx, dy in directions:
            new_i, new_j = i+dx, j+dy
            if new_i < m and new_j < n and new_i >= 0 and new_j >= 0 and (new_i, new_j) not in visited:
                if grid[new_i][new_j] == "1":
                    dfs(grid, new_i, new_j, visited)
            
    n_islands = 0
    visited = set()
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1" and (i, j) not in visited:
                dfs(grid, i, j, visited)
                n_islands += 1
    return n_islands


if __name__ == "__main__":
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    print(number_of_islands(grid))