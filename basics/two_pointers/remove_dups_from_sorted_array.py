"""
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place 
such that each unique element appears only once. The relative order of the elements should be kept the same.
Consider the number of unique elements in nums to be k​​​​​​​​​​​​​​. 
After removing duplicates, return the number of unique elements k.
The first k elements of nums should contain the unique numbers in sorted order. 
The remaining elements beyond index k - 1 can be ignored.
"""

def remove_dups_inplace(nums: list[int]) -> int:
    slow = 0 # write pos & the index of the last unique element
    # fist element is the last unique element
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]: # find a new element different than what we last wrote
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1 # the number of unique elements

if __name__ == "__main__":
    nums = [0,0,1,1,1,2,2,3,3,4]
    k = remove_dups_inplace(nums)
    print(nums[:k])