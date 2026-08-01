"""
Given an integer array nums, return an array answer such that answer[i] 
is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
"""


def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    prefix_prod_left = [1] * (n+1)
    for i in range(n):
        prefix_prod_left[i+1] = prefix_prod_left[i] * nums[i]

    prefix_prod_right = [1] * (n+1)
    for i in range(1, n+1):
        prefix_prod_right[n-i] = prefix_prod_right[n-i+1] * nums[n-i]

    result = []
    for i in range(n):
        result.append(prefix_prod_left[i] * prefix_prod_right[i+1])
    return result


if __name__ == "__main__":
    print(product_except_self(nums=[1,2,3,4]))