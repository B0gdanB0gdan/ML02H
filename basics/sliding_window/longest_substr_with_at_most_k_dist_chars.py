"""
You are given a string s consisting only lowercase alphabets and an integer k. 
Your task is to find the length of the longest substring that contains exactly k distinct characters.

Note : If no such substring exists, return -1. 

Input: s = "aabacbebebe", k = 3
Output: 7
Explanation: The longest substring with exactly 3 distinct characters is "cbebebe", which includes 'c', 'b', and 'e'.
"""
from collections import Counter


def longest_k_substr(s: str, k:int):
    left = 0
    best = -1
    state = Counter()
    distinct = 0
    for right, char in enumerate(s):
        state[char] += 1
        if state[char] == 1:
            distinct += 1

        while distinct > k:
            state[s[left]] -= 1
            if state[s[left]] == 0:
                distinct -= 1
            left += 1
        if distinct == k:
            best = max(best, right-left+1)

    return best

if __name__ == "__main__":
    print(longest_k_substr(s = "aabacbebebe", k = 3))