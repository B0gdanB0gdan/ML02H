"""
Given an unsorted array of integers nums, 
return the length of the longest consecutive elements sequence.
You must write an algorithm that runs in O(n) time.

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
"""


def longest_consecutive(nums: list[int]) -> int:
    num_set = set(nums)
    max_seq = 0
    for num in num_set:
        if num-1 not in num_set:
            current = num
            length = 1
            while current+1 in num_set:
                current = current + 1
                length += 1
            max_seq = max(max_seq, length)
    return max_seq


if __name__ == "__main__":
    print(longest_consecutive(nums=[0,3,7,2,5,8,4,6,0,1]))
    print(longest_consecutive(nums=[100,4,200,1,3,2]))