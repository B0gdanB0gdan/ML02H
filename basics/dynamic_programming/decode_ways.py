"""
1->A ... 26->Z

Input: digits = "121"
Output: 3
Explanation: The possible decodings are "ABA", "AU", "LA"

Input: digits = "1234"
Output: 3
Explanation: The possible decodings are "ABCD", "LCD", "AWD"
"""


def dw_memo(s, dp):
    """
    dp[s] = in how many ways we can decode substring s
    """

    # consumed everything
    if s == "":
        return 1
    
    if s[0] == "0":
        return 0

    if s in dp:
        return dp[s]

    dp[s] = dw_memo(s[1:], dp)
    if len(s) >= 2 and 10 <= int(s[:2]) <= 26:
        dp[s] += dw_memo(s[2:], dp)
    return dp[s]


def dw_tab(s):
    """
    dp[n] = in how many ways to decode the substring starting at index n
    Time: O(n)
    Space: O(n)
    """
    n = len(s)

    dp = [0] * (n + 1)

    # base case
    dp[n] = 1

    for i in range(n - 1, -1, -1):

        if s[i] != '0':
            dp[i] = dp[i+1]
        if i < n-1 and 10 <= int(s[i:i+2]) <= 26:
            dp[i] += dp[i+2]


    return dp[0]
    

def dw_tab_space_opt(s):
    """
    next1 = 
    next2 = 
    Time: O(n)
    Space: O(1)
    """
    n = len(s)
    next1, next2 = 1, 0

    for i in range(n - 1, -1, -1):

        temp = 0
        if s[i] != '0':
            temp += next1
        if i < n-1 and 10 <= int(s[i:i+2]) <= 26:
           temp += next2
        
        next2 = next1
        next1 = temp

    return next1


if __name__ == "__main__":
    s = "121"
    dp = {}
    print("Memo:", dw_memo(s, dp))
    print("Tab:", dw_tab(s))
    print("Tab Space O(1):", dw_tab_space_opt(s))