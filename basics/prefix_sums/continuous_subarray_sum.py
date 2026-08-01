"""
Given an integer array nums and an integer k, return true if nums has a good subarray or false otherwise.

A good subarray is a subarray where:
its length is at least two, and
the sum of the elements of the subarray is a multiple of k.
Note that:

A subarray is a contiguous part of the array.
An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.

Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.

Input: nums = [23,2,6,4,7], k = 13
Output: false
"""


def check_subarray_sum(nums: list[int], k: int) -> bool:
    if len(nums) < 2:
        return False
    
    prefix_sum = 0 # before the array
    seen = {0: -1}
    for i, num in enumerate(nums):
        prefix_sum += num
        rem = prefix_sum % k
        if rem in seen:
            if i - seen[rem] >= 2:
                return True
        else:
            seen[rem] = i

    return False


if __name__ == "__main__":
    print(check_subarray_sum(
        nums=[23,2,6,4,7],
        k = 6
    ))
    print(check_subarray_sum(
        nums=[23,2,6,4,7],
        k=13
    ))