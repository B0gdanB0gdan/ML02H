"""
Given two strings word1 and word2, return the minimum number of operations required 
to convert word1 to word2.

You have the following three operations permitted on a word:
Insert a character
Delete a character
Replace a character
 

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
"""


def min_edit_distance_memo_helper(word1, word2, m, n, dp):
    """
    dp[(m, n)] = min edit distance between substrings until m and n
    Time: O(m*n)
    Space: O(m*n)
    Space: 
    """
    if m == 0:
        return n
    
    if n == 0:
        return m
    
    if (m, n) in dp:
        return dp[(m, n)]

    if word1[m-1] == word2[n-1]:
        dp[(m, n)] =  min_edit_distance_memo_helper(word1, word2, m-1, n-1, dp)
    else:
        dp[(m, n)] = 1 + min(
            min_edit_distance_memo_helper(word1, word2, m-1, n-1, dp), # replace
            min_edit_distance_memo_helper(word1, word2, m, n-1, dp), # insertion
            min_edit_distance_memo_helper(word1, word2, m-1, n, dp) # deletion
        )
    return dp[(m, n)]

def min_edit_distance_memo(word1, word2):
    m = len(word1)
    n = len(word2)
    return min_edit_distance_memo_helper(word1, word2, m, n, {})


def min_edit_distance_tab(word1, word2):
    """
    dp[i][j] = edit distance of word1 until i and word2 until j
    """
    
    m = len(word1)
    n = len(word2)

    dp = [[0]*(n+1) for _ in range(m+1)] 

    for i in range(n):
        dp[i][0] = i

    for j in range(m):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], # deletion 
                    dp[i][j-1], # insertion
                    dp[i-1][j-1], # replace
                )
    return dp[m][n]

if __name__ == "__main__":
    word1 = "intention"
    word2 = "execution"
    print("Memo:", min_edit_distance_memo(word1, word2))
    print("Tab:", min_edit_distance_tab(word1, word2))