import argparse

"""
Input: s1 = "AGGTAB", s2 = "GXTXAYB"
Output: 4
Explanation: The longest common subsequence is "GTAB".
"""


# Recursive version
def lcs(s1, s2, m, n):
    
    if m == 0 or n==0:
        # traversal is complete
        if s1[m] == s2[n]:
            return 1
        return 0
	
    if s1[m] == s2[n]:
        return 1 + lcs(s1, s2, m-1, n-1)
    else:
        # first traversal of s1 as letters need to be in order
        return max(lcs(s1, s2, m-1, n), lcs(s1, s2, m, n-1))
    

def lcs_memo(s1, s2, m, n, memo=None):
    if memo is None:
        memo = {}   

    if m < 0 or n < 0:
        return 0
    
    if (m, n) in memo:
        return memo[(m, n)]
    
    if s1[m] == s2[n]:
        memo[(m, n)] = 1 + lcs_memo(s1, s2, m-1, n-1, memo)
    else:	
        memo[(m, n)] = max(lcs_memo(s1, s2, m-1, n, memo), lcs_memo(s1, s2, m, n-1, memo))
    
    return memo[(m, n)]


def lcs_tab(s1, s2, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def knapsack(w: list, v: list, c: int, n: int):
    # time O(2^n) because of the recursion tree, expands 2 every level
    # space O(n) because of the stack frames that exist simulataneously at the same time
    if n == 0 or c == 0:
        return 0

    pick = 0
    if w[n-1] <= c:
        pick = v[n-1] + knapsack(w, v, c - w[n-1], n-1)
    not_pick = knapsack(w, v, c, n-1)

    return max(pick, not_pick)


def knapsack_memo(w: list, v: list, c: int, n: int, memo=None):
    "What is the best value I can get using the first n items with remaining capacity c?"
    if memo is None:
        memo = {}

    if n == 0 or c == 0:
        return 0
    
    if (n, c) in memo:
        return memo[(n, c)] 

    pick = 0
    if w[n-1] <= c:
        pick = v[n-1] + knapsack(w, v, c - w[n-1], n-1, memo)
    not_pick = knapsack(w, v, c, n-1,memo)

    memo[(n, c)] = max(pick, not_pick)
    return memo[(n, c)]
	
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--s1', type=str, required=True)
    # parser.add_argument('--s2', type=str, required=True)

    w = [4, 5, 1]
    v = [1, 2, 3]
    c = 4
    print(knapsack(w, v, c, n=len(w)))
    
   
