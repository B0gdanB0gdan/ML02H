"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
such that i != j, i != k, and j != k, 
and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
"""

def three_sum(nums: list[int]):

    # fix one element nums[i] and find a, b where num[i] + a + b = 0

    nums.sort()
    result = []

    n = len(nums)
    for i in range(n-2): # n-3, n-2, n-1
        if i > 0  and nums[i] == nums[i-1]:
            continue
        left, right = i+1, n-1
        while left < right:
            s = nums[left] + nums[right] + nums[i]
            if s == 0:
                result.append([nums[left], nums[right], nums[i]])

                left += 1 # this pair of L,R already treated
                right -= 1
            
                # advance left until no duplication
                while left < right and nums[left] == nums[left-1]:
                    left += 1

                while left < right and nums[right] == nums[right+1]:
                    right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
                
    return result


if __name__ == "__main__":
    print(three_sum(
        nums=[-1,0,1,2,-1,-4]
    ))