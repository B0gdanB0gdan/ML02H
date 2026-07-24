"""
You are given an m x n integer array grid. 
There is a robot initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 or 0 respectively in grid. 
A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.
"""


def unique_paths2_memo_helper(obstacle_grid, m, n, dp):
    """
    dp[(m,n)] = unique paths reaching pos (m, n)
    Time: O(m*n) instead of 2^(m*n)    
    space: O(m*n)
    """ 

    if m < 0 or n < 0:
        return 0

    if m == 0 and n == 0:
        return 1
    
    if obstacle_grid[m][n] == 1:
        
        return 0 
    
    if (m,n) in dp:
        return dp[(m,n)]

    dp[(m, n)] = unique_paths2_memo_helper(obstacle_grid, m-1, n, dp) + unique_paths2_memo_helper(obstacle_grid, m, n-1, dp)
    return dp[(m, n)]

def unique_paths2_memo(obstacle_grid):
    m = len(obstacle_grid)
    n = len(obstacle_grid[0])
    return unique_paths2_memo_helper(obstacle_grid, m-1, n-1, {})


def unique_paths2_tab(obstacle_grid):
    """
    dp[(m,n)] = unique paths reaching pos (m, n)
    """
    m = len(obstacle_grid)
    n = len(obstacle_grid[0])
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        if obstacle_grid[i][0] == 0:
            dp[i][0] = 1 if i == 0 else dp[i-1][0]
    
    for j in range(n):
        if obstacle_grid[0][j] == 0:
            dp[0][j] = 1 if j == 0 else dp[0][j-1]

    for i in range(1, m):
        for j in range(1, n):
            if obstacle_grid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

if __name__ == "__main__":
    obstacle_grid = [[0,0,0],[0,1,0],[0,0,0]]
    print("Memo:", unique_paths2_memo(obstacle_grid))
    print("Tab:", unique_paths2_tab(obstacle_grid))
    obstacle_grid = [[0,1,0]]
    print("Memo:", unique_paths2_memo(obstacle_grid))
    print("Tab:", unique_paths2_tab(obstacle_grid))