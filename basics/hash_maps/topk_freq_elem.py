"""
Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
"""
from collections import Counter


def topk_frequent(nums: list[int], k: int) -> list[int]:
    topk = Counter(nums).most_common(k)
    return [item[0] for item in topk]


if __name__ == "__main__":
    print(topk_frequent([1,1,1,2,2,3], k=2))