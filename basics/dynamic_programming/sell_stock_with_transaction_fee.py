"""
You are given an array prices where prices[i] is the price of a given stock on the ith day, 
and an integer fee representing a transaction fee.

Find the maximum profit you can achieve. 
You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

Note:

You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
The transaction fee is only charged once for each stock purchase and sale.
 

Example 1:

Input: prices = [1,3,2,8,4,9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
- Buying at prices[0] = 1
- Selling at prices[3] = 8
- Buying at prices[4] = 4
- Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
Example 2:

Input: prices = [1,3,7,5,10,3], fee = 3
Output: 6
"""


def profit_fee_memo_helper(prices, fee, n, bought, dp):
    """
    dp[n][bought] = max profit until day n for a given state bought (either True or False)
    Time: O(n) otherwise 2^n
    Space: O(n)
    """
    if n == 0:
        return 0
    
    if (n, bought) in dp:
        return dp[(n, bought)]
    
    if bought == False:
        dp[(n, False)] = max(
            -prices[n-1] + profit_fee_memo_helper(prices, fee, n-1, True, dp),
            profit_fee_memo_helper(prices, fee, n-1, False, dp)
        )
    else:
        dp[(n, True)] = max(
            prices[n-1] - fee + profit_fee_memo_helper(prices, fee, n-1, False, dp), 
            profit_fee_memo_helper(prices, fee, n-1, True, dp)
        )
    return dp[(n, bought)]


def profit_fee_memo(prices, fee):
    n = len(prices)
    bought = False
    return profit_fee_memo_helper(prices, fee, n, bought, {})


def profit_fee_tab(prices, fee):
    """
    dp[n][bought] = max profit until day n for a given state bought (either True or False)
    Time: O(n) otherwise 2^n
    Space: O(n)
    """
    n = len(prices)
    dp = [[0]*(2) for _ in range(n+1)]
    
    for i in range(1, n+1):
        dp[i][0] = max(
            -prices[i-1] + dp[i-1][1],
            dp[i-1][0]
        )
        dp[i][1] = max(
            prices[i-1] - fee + dp[i-1][0],
            dp[i-1][1]
        )
    
    return dp[n][0]



if __name__ == "__main__":
    prices = [1,3,2,8,4,9]
    fee = 2
    print("Memo:", profit_fee_memo(prices, fee))
    print("Tab:", profit_fee_tab(prices, fee))