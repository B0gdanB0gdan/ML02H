"""
Given a m x n grid filled with non-negative numbers, 
find a path from top left to bottom right, 
which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
"""


def min_path_sum_memo_helper(grid, m, n, dp):
    """
    dp[(m,n)] = min cost of reaching state m, n
    Time: O(m*n) 
    space: O(m*n)
    """

    if m == 0 and n == 0:
        return grid[0][0]
    
    if (m,n) in dp:
        return dp[(m,n)]

    up, left = float('inf'), float('inf')
    if m-1 >= 0:
        up = min_path_sum_memo_helper(grid, m-1, n, dp)
    if n-1 >= 0:
        left = min_path_sum_memo_helper(grid, m, n-1, dp)

    dp[(m, n)] = grid[m][n] + min(up, left)

    return dp[(m, n)]

def min_path_sum_memo(grid):
    m = len(grid)
    n = len(grid[0])
    return min_path_sum_memo_helper(grid, m-1, n-1, {})


def min_path_sum_tab(grid):
    m = len(grid)
    n = len(grid[0])
    dp = [[0] * n for _ in range(m)]

    dp[0][0] = grid[0][0]

    for i in range(m):
        for j in range(n):
            dp[i][j] = grid[i][j]
            if i == 0 and j == 0:
                continue
            up, left = float('inf'), float('inf')
            if i > 0:
                up = dp[i-1][j]
            if j > 0:
                left = dp[i][j-1]
            dp[i][j] += min(up, left)

    return dp[m-1][n-1]

if __name__ == "__main__":
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    print("Memo:", min_path_sum_memo(grid))
    print("Tab:", min_path_sum_tab(grid))
    rid = [[1,2,3],[4,5,6]]
    print("Memo:", min_path_sum_memo(grid))
    print("Tab:", min_path_sum_tab(grid))