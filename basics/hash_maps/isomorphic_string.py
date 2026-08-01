"""
Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving 
the order of characters. No two characters may map to the same character, but a character may map 
to itself.

Input: s = "egg", t = "add"
Output: true

Input: s = "f11", t = "b23"

Output: false
Explanation:
The strings s and t can not be made identical as '1' needs to be mapped to both '2' and '3'.
"""

from collections import Counter


def is_isomorphic(s: str, t: str) -> bool:

    n = len(s)
    iso_st, iso_ts = {}, {}
    for i in range(n):
        if s[i] not in iso_st:
            iso_st[s[i]] = t[i]
        if t[i] not in iso_ts:
            iso_ts[t[i]] = s[i]
        if t[i] != iso_st[s[i]] or s[i] != iso_ts[t[i]]:
            return False
    return True


if __name__ == "__main__":
    print(is_isomorphic("egg", "add"))
    print(is_isomorphic("f11", "b23"))
    print(is_isomorphic("ab", "aa"))