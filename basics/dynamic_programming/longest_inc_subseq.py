"""
Given an integer array nums, return the length of the longest strictly increasing subsequence.

Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4
"""

"""

"""


def lis_memo_helper(nums, i, dp):
    """
    dp[i] = length of longest subseq ending at nums[i]
    Time: O(n*n)
    Space: O(n)
    """
    if i == 0:
        return 1
    
    if dp[i] != -1:
        return dp[i]

    max_len = 1
    for k in range(i):
        if nums[i] > nums[k]:
            max_len = max(max_len, 1 + lis_memo_helper(nums, k, dp))
    dp[i] = max_len
    return dp[i]

def lis_memo(nums):
    n = len(nums)
    max_len = 0
    dp = [-1] * n
    for i in range(n):
        max_len = max(max_len, lis_memo_helper(nums, i, dp))
    return max_len

def lis_tab(nums):
    """
    dp[i] = length of longest subseq ending at nums[i]
    Time: O(n*n)
    Space: O(n)
    """
    
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        max_len = 1
        for k in range(i):
            if nums[i] > nums[k]:
                max_len = max(max_len, dp[k] + 1)
        dp[i] = max_len

    return max(dp)

if __name__ == "__main__":
    nums = [10,9,2,5,3,7,101,18]
    print("Memo:", lis_memo(nums))
    print("Tab:", lis_tab(nums))
    nums = [0,1,0,3,2,3]
    print("Memo:", lis_memo(nums))
    print("Tab:", lis_tab(nums))
    nums = [7,7,7,7,7,7,7]
    print("Memo:", lis_memo(nums))
    print("Tab:", lis_tab(nums))
    nums = [5, 6, 5, 2, 3]
    print("Memo:", lis_memo(nums))
    print("Tab:", lis_tab(nums))
    nums = [5, 4, 4, 5]
    print("Memo:", lis_memo(nums))
    print("Tab:", lis_tab(nums))