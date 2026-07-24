"""
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. 
Once you pay the cost, you can either climb one or two steps.
You can either start from the step with index 0, or the step with index 1.
Return the minimum cost to reach the top of the floor.
"""


def min_cs_memo(n, cost, dp):
    """
    Time: O(n)
    Space: O(n)
    dp[i] = min cost to climb i remaining stairs
    """
    if n <= 1:
        return cost[n]
    
    if dp[n] != -1:
        return dp[n]
    
    dp[n] = cost[n] + min(min_cs_memo(n-1, cost, dp), min_cs_memo(n-2, cost, dp))
    return dp[n]

def min_cs_tab(cost):
    """
    Time: O(n)
    Space: O(n)
    dp[i] = min cost to climb i remaining stairs
    """
    n = len(cost) - 1 # already appended 0
    dp = [0] * (n+1)
    dp[0] = cost[0]
    dp[1] = cost[1]
    for i in range(2, n+1):
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    return dp[n]


if __name__ == "__main__":
    cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    n = len(cost)
    cost.append(0)
    dp = [-1] * (n+1)
    print("Memoization:", min_cs_memo(n, cost, dp))
    print("Tabulation:", min_cs_tab(cost))