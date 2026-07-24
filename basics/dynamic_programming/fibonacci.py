

def fibonacci_memo(n: int, dp: list):
    """
    Time: O(n)
    Space: O(n)
    """
    if n <= 1:
        return n

    if dp[n] != -1:
        return dp[n]
    
    fib_n = fibonacci_memo(n-1, dp) + fibonacci_memo(n-2, dp)

    dp[n] = fib_n
    return dp[n]


def fibonacci_tab(n):
    """
    Time: O(n)
    Space: O(n)
    """
    dp = [-1] * (n+1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]



if __name__ == "__main__":
    n = 5
    dp = [-1] * (n+1)
    print("Memoization:", fibonacci_memo(n, dp))
    print("Tabulation:", fibonacci_tab(n))