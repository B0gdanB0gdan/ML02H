"""
Given an integer array nums, find a subarray that has the largest product, and return the product.
The test cases are generated so that the answer will fit in a 32-bit integer.
Note that the product of an array with a single element is the value of that element.

Example 1:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
"""

def mps_memo_helper(nums, dp, i):
    """
    dp_min[i] = max product of subarray ending at idx i
    """
    if i == 0:
        dp[i] = (nums[0], nums[0])
        return dp[i]

    if i in dp:
        return dp[i]
    
    prev_max, prev_min = mps_memo_helper(nums, dp, i-1)
    prev_max = max(nums[i], nums[i]*prev_max, nums[i]*prev_min)
    prev_min = min(nums[i], nums[i]*prev_max, nums[i]*prev_min)
    dp[i] = (prev_max, prev_min)
    return dp[i]


def mps_memo(nums):
    n = len(nums)
    dp = {}
    mps_memo_helper(nums, dp, n-1)
    return max(dp[i][0] for i in range(n))


def mps_tab(nums):
    res = prev_max = prev_min = nums[0] 
    for n in nums[1:]:
        prev_max = max(n, n*prev_max, n*prev_min)
        prev_min = min(n, n*prev_max, n*prev_min)
        res = max(res, prev_max)
    return res


if __name__ == "__main__":
    nums = [2,3,-2,4]
   
    print("Memo:", mps_memo(nums))
    print("Tab:", mps_tab(nums))