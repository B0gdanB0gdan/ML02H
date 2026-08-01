"""
You are given an integer array nums consisting of n elements, and an integer k.
Find a contiguous subarray whose length is equal to k that has the maximum average value 
and return this value. Any answer with a calculation error less than 10^-5 will be accepted.

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
Example 2:

Input: nums = [5], k = 1
Output: 5.00000
"""

def max_avg_subarray(nums: list[int], k: int):
    win_sum = sum(nums[:k])
    max_sum = win_sum
    n = len(nums)
    for i in range(k, n):
        win_sum += nums[i] - nums[i-k]
        max_sum = max(win_sum, max_sum)
    return max_sum / k


if __name__ == "__main__":
    print(max_avg_subarray(
        nums=[1,12,-5,-6,50,3],
        k=4
    ))
    print(max_avg_subarray(
        nums=[5],
        k = 1
    ))