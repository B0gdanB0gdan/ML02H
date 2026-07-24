"""
You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed. 
All houses at this place are arranged in a circle. 
That means the first house is the neighbor of the last one. 
Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
"""

from house_robber import house_robber_memo


def house_robber_memo2(arr):
    """
    dp[n] = max amount of money we can rob from n remaining house
    """
    dp1 = [-1] * (len(arr) - 1)
    dp2 = [-1] * (len(arr) - 1)
    return max(arr[0], house_robber_memo(arr[:-1], dp1), house_robber_memo(arr[1:], dp2))


def house_robber_tab_helper(arr):
    """
    Time: O(n)
    Space: O(1)
    """

    # we need the 2 max values of 2 possible ways

    rob1, rob2 = 0, 0
    # [rob1, rob2, h3, h4, h5, ...]
    for v in arr:
        temp = max(v + rob1, rob2)
        rob1 = rob2
        rob2 = temp

    return rob2



def house_robber_tab2(arr):
    return max(arr[0], house_robber_tab_helper(arr[1:]), house_robber_tab_helper(arr[:-1]))


if __name__ == "__main__":
    arr = [5, 3, 4, 11, 2]
    print("Memo:", house_robber_memo2(arr))
    print("Tab:", house_robber_tab2(arr))