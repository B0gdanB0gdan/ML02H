"""
Given a string s, find the length of the longest substring without duplicate characters.


Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3. Note that "bca" and "cab" are also correct answers.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
"""

from collections import Counter


def longest_substr_wo_rep_chars(s: str):
    left = 0
    win_state = Counter()
    best = 0

    def is_valid(state):
        return all(count <= 1 for count in state.values())

    for right in range(len(s)):
        win_state[s[right]] += 1

        while not is_valid(win_state):
            win_state[s[left]] -= 1
            if win_state[s[left]] == 0:
                del win_state[s[left]]
            left += 1

        best = max(best, right-left+1)
    return best

def longest_substr_wo_rep_chars_v2(s: str):
    left = 0
    window_state = Counter()
    duplicates = 0
    best = 0

    for right in range(len(s)):
        window_state[s[right]] += 1

        if window_state[s[right]] == 2:
            duplicates += 1

        while duplicates > 0:
            window_state[s[left]] -= 1
            if window_state[s[left]] == 1:
                duplicates -= 1
            left += 1
        best = max(best, right-left+1)

    return best


def longest_substr_wo_rep_chars_v3(s: str):
    seen = dict()
    left = 0
    best = 0
    for right, char in enumerate(s):
        if char in seen:
            left = seen[char] + 1
        seen[char] = right
        best = max(best, right-left+1)

    return best


if __name__ == "__main__":
    print(longest_substr_wo_rep_chars("abcabcbb"))
    print(longest_substr_wo_rep_chars_v2("abcabcbb"))
    print(longest_substr_wo_rep_chars_v3("abcabcbb"))