"""
Given an array arr[] where each element shows the amount of money in a row of houses built in a line. 
A robber wants to steal money, but cannot rob two houses adjacent to each other because it will set off an alarm.
Find the maximum money the robber can steal without robbing two adjacent houses.
"""

def house_robber_memo(arr, dp):
    """
    dp[n] = max amount of money we can rob from n remaining house
    """
    n = len(arr)-1

    if n < 0:
        return 0

    if dp[n] != -1:
        return dp[n]

    dp[n] = max(arr[0] + house_robber_memo(arr[2:], dp), house_robber_memo(arr[1:], dp))
    return dp[n]

def house_robber_tab(arr):
    n = len(arr)
    dp = [0] * (n+2)
    
    for i in range(n, 0, -1):
        j = i-1
        dp[j] = max(arr[j] + dp[j+2], dp[j+1])

    return dp[0]


if __name__ == "__main__":
    arr = [5, 3, 4, 11, 2]
    dp = [-1] * len(arr)
    print("Memo:", house_robber_memo(arr, dp))
    print("Tab:", house_robber_tab(arr))