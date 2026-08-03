"""
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
 
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Input: nums = [0]
Output: [[],[0]]
"""

def subsets2(nums: list):
    # duplicates
    # idea: sort + skip adjacent values if duplicates
    result = []
    nums.sort()
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i+1, path)
            path.pop()

    backtrack(0, [])
    return result


if __name__ == "__main__":
    nums = [1,2,2]
    print(subsets2(nums))