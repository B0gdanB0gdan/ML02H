"""
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.


Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
"""


def cs_memo(coins, am, dp):
    
    if am == 0:
        return 0
    
    if am < 0:
        return float("inf")
    
    if am in dp:
        return dp[am]

    min_next = float("inf")
    for c in coins:
        min_next = min(min_next, 1+cs_memo(coins, am-c, dp))
    dp[am] = min_next
    return dp[am]


def cs_tab(coins, am):
    """
    dp[i] = min amount of coins to build sum i
    """
    dp = [0] * (am+1)
    for s in range(1, am+1):
        min_coins = float("inf")
        for c in coins:
            if s-c >= 0:
                min_coins = min(min_coins, 1+dp[s-c])
        dp[s] = min_coins

    return dp[am]


if __name__ == "__main__":
    coins = [1,2,5]
    am = 11
    dp = {}
    print("Memo", cs_memo(coins, am, dp))
    print("Tab", cs_tab(coins, am))