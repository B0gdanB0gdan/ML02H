"""
There is a robot on an m x n grid. 
The robot is initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.

Given the two integers m and n, 
return the number of possible unique paths that the robot can take 
to reach the bottom-right corner.
"""


def unique_paths_memo_helper(m, n, dp):
    """
    dp[(m,n)] = unique paths reaching pos (m, n)
    Time: O(m*n) instead of 2^(m*n)    
    space: O(m*n)
    """
    if m == 1 or n == 1:
        return 1
    
    if (m,n) in dp:
        return dp[(m,n)]

    dp[(m, n)] = unique_paths_memo_helper(m-1, n, dp) + unique_paths_memo_helper(m, n-1, dp)
    return dp[(m, n)]

def unique_paths_memo(m, n):
    return unique_paths_memo_helper(m, n, {})


def unique_paths_tab(m, n):
    
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        dp[i][0] = 1
    
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

if __name__ == "__main__":
    print("Memo:", unique_paths_memo(m=3, n=7))
    print("Tab:", unique_paths_tab(m=3, n=7))

    print("Memo:", unique_paths_memo(m=3, n=2))
    print("Tab:", unique_paths_tab(m=3, n=2))