"""
Given two strings s and t, return true if they are equal when 
both are typed into empty text editors. 
'#' means a backspace character.
Note that after backspacing an empty text, the text will continue empty.

Example 1:
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".

Example 2:
Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".

Example 3:
Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".
"""

def backspace_compare(s: str, t: str) -> bool:
    i = len(s) - 1
    j = len(t) - 1

    skip_s = 0
    skip_t = 0

    while i >= 0 or j >= 0:

        while i >= 0:
            if s[i] == '#':
                skip_s += 1
                i -= 1
            elif skip_s > 0:
                skip_s -= 1
                i -= 1
            else:
                break

        while j >= 0:
            if t[j] == '#':
                skip_t += 1
                j -= 1
            elif skip_t > 0:
                skip_t -= 1
                j -= 1
            else:
                break

        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            return False

        i -= 1
        j -= 1

    return True


if __name__ == "__main__":
    print(backspace_compare("ab#c", "ad#c"))
    print(backspace_compare("ab##", "c#d#"))
    print(backspace_compare("a#c", "b"))