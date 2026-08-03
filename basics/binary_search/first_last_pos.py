"""
Given an array of integers nums sorted in non-decreasing order, 
find the starting and ending position of a given target value.
If target is not found in the array, return [-1, -1].
You must write an algorithm with O(log n) runtime complexity.

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Input: nums = [], target = 0
Output: [-1,-1]
"""

def search_range(nums: list[int], target: int) -> list[int]:
    left, right = 0, len(nums)-1

    first_occ = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            first_occ = mid
            right = mid - 1
        elif target < nums[mid]:
            right = mid-1
        else:
            left = mid + 1

    left, right = 0, len(nums)-1
    last_occ = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            last_occ = mid
            left = mid+1
        elif target < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return [first_occ, last_occ]

if __name__ == "__main__":
    print(search_range([5,7,7,8,8,10], target = 8))