"""
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that
every character in t (including duplicates) is included in the window. 
If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
"""
from collections import Counter


def min_window_substring(s, t):
    left = 0
    best = len(s)

    need = Counter(t)
    missing = len(t)


    for right, char in enumerate(s):

        if char in need:
            need[char] -= 1 # allow to get negative for surplus
            if need[char] == 0:
                missing -= 1

        while missing == 0:
            best = min(best, right-left+1)

            if s[left] in need:
                need[s[left]] += 1
                if need[s[left]] > 0:
                    missing += 1
            left += 1

    return best



if __name__ == "__main__":
    print(min_window_substring(
        s="ADOBECODEBANC",
        t="ABC")
    )