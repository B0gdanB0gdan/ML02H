"""
Given two strings, s1 and s2, find the length of the Longest Common Subsequence. 
If there is no common subsequence, return 0. 
A subsequence is a string generated from the original string by deleting 0 or more characters, without changing the relative order of the remaining characters.

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
    """
    memo[(m, n)] = max subseq len for first m from s1 and first n from s2
    """
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
    """
    dp[i][j] = max subseq len for first m from s1 and first n from s2
    """

    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            l = 0
            if s1[i-1] == s2[j-1]:
                l = 1 + dp[i-1][j-1] 
            else:
                l = max(dp[i][j-1], dp[i-1][j])
            dp[i][j] = l

    return dp[m][n]


if __name__ == "__main__":
    s1 = "AGGTAB"
    s2 = "GXTXAYB"
    m, n = len(s1), len(s2)
    print("Memo:", lcs_memo(s1, s2, m-1, n-1))
    print("Tab:", lcs_tab(s1, s2, m, n))