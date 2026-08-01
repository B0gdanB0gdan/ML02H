"""
Given two strings ransomNote and magazine, return true if ransomNote can be constructed 
by using the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

Example 1:
Input: ransomNote = "a", magazine = "b"
Output: false
Example 2:
Input: ransomNote = "aa", magazine = "ab"
Output: false
Example 3:
Input: ransomNote = "aa", magazine = "aab"
Output: true
"""

from collections import Counter


def can_construct(ransom_note: str, magazine: str) -> bool:
    counter_ransom = Counter(ransom_note)
    for l in magazine:
        if counter_ransom[l] > 0:
            counter_ransom[l] -= 1
    return counter_ransom.total() == 0


if __name__ == "__main__":
    print(can_construct("a", "b"))
    print(can_construct("aa", "ab"))
    print(can_construct("aa", "aab"))