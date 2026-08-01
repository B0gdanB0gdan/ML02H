"""
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters 
and removing all non-alphanumeric characters, it reads the same forward and backward.
Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.
"""

def is_palindrome(s: str):
    s = s.lower()
    if s == "":
        return True
    
    left, right = 0, len(s) - 1
    while left < right:

        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left] != s[right]:
            return False

        left += 1
        right -=1

    return True


if __name__ == "__main__":
    print(is_palindrome(s = "A man, a plan, a canal: Panama"))