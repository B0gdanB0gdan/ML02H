"""
Given the dimension of a sequence of matrices in an array arr[], 
where the dimension of the ith matrix is (arr[i-1] * arr[i]), 
the task is to find the most efficient way to multiply these matrices together such that
the total number of element multiplications is minimum. 
When two matrices of size m*n and n*p when multiplied, 
they generate a matrix of size m*p and the number of multiplications performed is m*n*p.

Examples:

Input: arr[] = [2, 1, 3, 4]
Output: 20
Explanation: There are 3 matrices of dimensions 2x1, 1x3, and 3x4, 
Let the input 3 matrices be M1, M2, and M3. 
There are two ways to multiply ((M1 x M2) x M3) and (M1 x (M2 x M3)), 
Please note that the result of M1 x M2 is a 2 x 3 matrix and result of (M2 x M3) is a 1 x 4 matrix.

((M1 x M2) x M3)  requires (2 x 1 x 3) + (2 x 3 x 4) = 30 
(M1 x (M2 x M3))  requires (1 x 3 x 4) + (2 x 1 x 4) = 20 
The minimum of these two is 20.
"""

def matrix_chain_memo_helper(arr, i, j, dp):
    """
    dp[(i,j)] = minimum cost to fully reduce matrices Mi ... Mj into a single matrix.
    Time: O(n*n*n) instead of 2^n without dp
    Space: O(n*n)
    """
    if i+1 == j:
        # single matrix, everything is reduced
        return 0
    
    if (i, j) in dp:
        return dp[(i, j)]
    
    min_ops = float("inf")
    for k in range(i+1, j):
        ops = matrix_chain_memo_helper(arr, i, k, dp) + matrix_chain_memo_helper(arr, k, j, dp) + arr[i] * arr[k] * arr[j]
        min_ops = min(min_ops, ops)
    dp[(i, j)] = min_ops
    return dp[(i, j)]
        

def matrix_chain_memo(arr):
    n = len(arr)
    i = 0
    j = n - 1
    return matrix_chain_memo_helper(arr, i, j, {})


def matrix_chain_tab(arr):
    """
    dp[(i,j)] = minimum cost to fully reduce matrices Mi ... Mj into a single matrix.
    Time: O(n*n*n)
    Space: O(n*n)
    """
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            dp[i][j] = float('inf')

            for k in range(i+1, j):
                cost = dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


if __name__ == "__main__":
    arr = [2, 1, 3, 4]
    print("Memo:", matrix_chain_memo(arr))
    print("Tab:",  matrix_chain_tab(arr))


