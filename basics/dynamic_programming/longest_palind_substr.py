"""
Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"

Example 2:
Input: s = "cbbd"
Output: "bb"

Brute force: O(n^3)
"""

def lps_memo_helper(s, dp, i, j):
    
    if i >= j:
        return True

    if dp[i][j] is True:
        return dp[i][j]

    dp[i][j] = (s[i] == s[j]) and lps_memo_helper(s, dp, i+1, j-1)
    return dp[i][j]

def lps_memo(s):
    """
    dp[i][j] = True if s[i:j+1] is a palindrome
    Space: O(n^2)
    Time: O(n^2)
    """

    n = len(s)
    dp = [[False] * n for _ in range(n)]

    best = ""
    for i in range(n):
        for j in range(i, n):

            substr = s[i:j+1]
            if lps_memo_helper(s, dp, i, j):
                if len(substr) > len(best):
                    best = substr

    return best
   

def lps_tab(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    
    best = s[0] if s else ""

    # length 1 palindromes
    for i in range(n):
        dp[i][i] = True

    # length 2 palindromes
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True 
            best = s[i:i+2]

    for length in range(3, n+1):
        for i in range(0, n-length+1):
            j = i + length - 1
            dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1]
            if dp[i][j] and len(s[i:j+1]) > len(best):
                best = s[i:j+1]

    return best
    



if __name__ == "__main__":
    s = "cbbd"
    
    print("Memo:", lps_memo(s))
    print("Tab:", lps_tab(s))