"""
Given two arrays, v[] and w[], where each element represents the value and weight of an item respectively, 
and an integer c representing the maximum capacity of the knapsack.

The task is to put the items into the knapsack such that 
the total value obtained is maximum without exceeding the capacity W.

Input: c = 4, v[] = [1, 2, 3], w[] = [4, 5, 1]
Output: 3
Explanation: Choose the last item, which weighs 1 unit and has a value of 3.
"""


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
    """
    memo[(n, c)] = best value we can get with remaining n items and capacity c
    Time: O(n x c)
    Space: O(n x c)
    """
    
    if memo is None:
        memo = {}

    if n == 0 or c == 0:
        return 0
    
    if (n, c) in memo:
        return memo[(n, c)] 

    pick = 0
    if w[n-1] <= c:
        pick = v[n-1] + knapsack_memo(w, v, c - w[n-1], n-1, memo)
    not_pick = knapsack_memo(w, v, c, n-1,memo)

    memo[(n, c)] = max(pick, not_pick)
    return memo[(n, c)]


def knapsack_tab(w: list, v: list, c: int):
    """
    dp[i][j] = max value obtained using first i items with a capacity of j.
    """
    n = len(w)
    dp = [[0 for _ in range(c + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(c + 1):

            if i == 0 or j == 0:
                dp[i][j] = 0
            else:
                pick = 0

                if w[i - 1] <= j:
                    pick = v[i - 1] + dp[i - 1][j - w[i - 1]]

                notPick = dp[i - 1][j]

                dp[i][j] = max(pick, notPick)

    return dp[n][c]



if __name__ == "__main__":
    w = [4, 5, 1]
    v = [1, 2, 3]
    c = 4
    print("Memo:", knapsack_memo(w, v, c, n=len(w)))
    print("Tab:", knapsack_tab(w, v, c))