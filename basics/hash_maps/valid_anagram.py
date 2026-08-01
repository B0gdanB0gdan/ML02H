"""Given two strings s and t, return true if t is an anagram of s, and false otherwise."""
from collections import Counter


def is_anagram(s, t):
    """
    Time: O(nlogn)
    Space: O(n)
    """
    return sorted(s) == sorted(t)


def is_anagram2(s, t):
    """
    Time: O(n)
    Space: O(n)
    """
    return Counter(s) == Counter(t)

if __name__ == "__main__":
    s = "anagram"
    t = "nagaram"
    print(is_anagram(s, t))
    print(is_anagram2(s, t))
