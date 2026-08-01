"""
Given an array nums with n objects colored red, white, or blue, 
sort them in-place so that objects of the same color are adjacent, 
with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.
"""


def sort_colors(nums: list[int]):
    n = len(nums)
    left, right = 0, n-1

    # left is write pos for 0s
    # right is write pos for 2s
    mid = 0
    while mid <= right:
        if nums[mid] == 0:
            nums[mid], nums[left] = nums[left], nums[mid]
            left += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[right] = nums[right], nums[mid]
            right -= 1

    return nums


if __name__ == "__main__":
    print(sort_colors(nums = [1,0]))
    print(sort_colors(nums = [2,0,1]))
    print(sort_colors(nums = [0,2,1,2]))
    print(sort_colors(nums = [2,0,2,1,1,0]))