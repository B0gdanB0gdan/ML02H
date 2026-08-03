"""
Given an array nums of distinct integers, return all the possible permutations.
You can return the answer in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]
"""


def permutations(nums: list[int]):

    result = []
    def backtrack(used, path):
        if len(path) == len(nums):
            result.append(path[:])

        for i in range(len(nums)):
            if used[i]:
                continue
           
            path.append(nums[i])
            used[i] = True
            backtrack(used, path)
            path.pop()
            used[i] = False

    backtrack([False]*len(nums), [])
    return result


if __name__ == "__main__":
    nums = [1,2,3]
    print(permutations(nums))