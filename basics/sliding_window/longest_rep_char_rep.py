"""
You are given a string s and an integer k. 
You can choose any character of the string and change it to any other uppercase English character. 
You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too.
"""

from collections import Counter


def character_replacement(s: str, k: int):
    left = 0
    state = Counter()
    best = 0
    max_freq = 0
    for right, char in enumerate(s):
        state[char] += 1

        max_freq = max(max_freq, state[char])
        win_size = right-left + 1

        if win_size - max_freq > k:
            state[s[left]] -= 1
            left += 1

        best = max(best, right-left+1)

    return best
        

if __name__ == "__main__":
    print(character_replacement(s = "ABAB", k = 2))
    print(character_replacement(s = "AABABBA", k = 1))