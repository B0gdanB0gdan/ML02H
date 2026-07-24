"""
Given an integer n, return the number of structurally unique BST's (binary search trees) 
which has exactly n nodes of unique values from 1 to n.

Example:
Input: n = 3
Output: 5
"""
    

def num_trees_memo_helper(n, dp):
    """
    dp[n] = number of unique BSTs
    Time: O(n*n)
    Space: O(n)
    """
    if n == 0:
        return 1
    
    if dp[n] != -1:
        return dp[n]
    
    res = 0
    for root in range(1, n+1):
        res += num_trees_memo_helper(root-1, dp) * num_trees_memo_helper(n-root, dp)
    dp[n] = res
    return dp[n]

def num_trees_memo(n):
    dp = [-1] * (n+1)
    return num_trees_memo_helper(n, dp)

def num_trees_tab(n):
    dp = [1] * (n+1)

    for i in range(1, n+1):
        res = 0
        for root in range(1, i+1):
            res += dp[root-1] * dp[i-root]
        dp[i] = res
    return dp[n]



if __name__ == "__main__":
    n = 3
    print("Memo:", num_trees_memo(n))
    print("Tab:", num_trees_tab(n))