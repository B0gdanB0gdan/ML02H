"""
Given two strings s and t, return the number of distinct subsequences of s which equals t.

Input: s = "rabbbit", t = "rabbit"
Output: 3

Input: s = "babgbag", t = "bag"
Output: 5
"""

def distinct_subseq_memo_helper(s, t, m, n, dp):
    """
    dp[m][n] = number of distinct subseq of s[:m] which equal t[:n]
    Time: O(m*n)
    Space: O(m*n)
    """
    if n == 0:
        return 1
    
    if m == 0:
        return 0
    
    if dp[m-1][n-1] != -1:
        return dp[m-1][n-1]
    
    if s[m-1] == t[n-1]:
        dp[m-1][n-1] = distinct_subseq_memo_helper(s, t, m-1, n-1, dp) + distinct_subseq_memo_helper(s, t, m-1, n, dp)
    else:
        dp[m-1][n-1] = distinct_subseq_memo_helper(s, t, m-1, n, dp)
    return dp[m-1][n-1]


def distinc_subseq_memo(s, t):
    m = len(s)
    n = len(t)
    dp = [[-1]*n for _ in range(m)]
    return distinct_subseq_memo_helper(s, t, m, n, dp)

def distinct_subseq_tab(s, t):
    """
    dp[m][n] = number of distinct subseq of s[:m] which equal t[:n]
    Time: O(m*n)
    Space: O(m*n)
    """
    m = len(s)
    n = len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = 1

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]

    return dp[m][n]


if __name__ == "__main__":
    s = "babgbag"
    t = "bag"
    print("Memo:",  distinc_subseq_memo(s, t))
    print("Tab:", distinct_subseq_tab(s, t))