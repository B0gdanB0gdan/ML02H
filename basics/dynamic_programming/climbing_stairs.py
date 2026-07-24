"""
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps.
In how many distinct ways can you climb to the top?
"""

def cs_memo(n, dp):
    """
    Time: O(n)
    Space: O(n)
    dp[i] = number of distinct ways to climb i remaining stairs
    """
    if n <= 1:
        return 1
    if dp[n] != -1:
        return dp[n]
    dp[n] = cs_memo(n-1, dp) + cs_memo(n-2, dp)
    return dp[n]

def cs_tab(n):
    """
    Time: O(n)
    Space: O(n)
    dp[i] = number of distinct ways to climb i remaining stairs
    """
    dp = [0] * (n+1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]


if __name__ == "__main__":
    n = 5
    dp = [-1] * (n+1)
    print("Memoization:", cs_memo(n, dp))
    print("Tabulation:", cs_tab(n))