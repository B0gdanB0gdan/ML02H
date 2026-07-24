"""
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into 
a space-separated sequence of one or more dictionary words.
Note that the same word in the dictionary may be reused multiple times in the segmentation.

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
"""


def wb_memo_helper(s: str, i: int, word_dict: list, dp: dict):
    """
    dp[i] = if substring starting at i can be decomposed 
    Time: O(n*w*l), l = len of the word
    Space: O(n) or O(2*n) because of dp+recursion stack frames
    """
    if i >= len(s):
        return True
    
    if i in dp:
        return dp[i]

    dp[i] = False
    for word in word_dict:
        wl = len(word)
        if s[i:i+wl] == word:
            if wb_memo_helper(s, i+wl, word_dict, dp):
                dp[i] = True
                break
    return dp[i]


def wb_memo(s: str, word_dict: list):
    return wb_memo_helper(s, 0, word_dict, {})


def wb_tab(s, word_dict):
    """
    dp[i] = substring until index i can be split into words
    Time: O(n*w*l)
    Space: O(n) but no possible RecursionError
    """

    n = len(s)
    dp = [False] * (n+1)
    dp[0] = True
    for i in range(n+1):
        for word in word_dict:
            wl = len(word)
            if i >= wl and s[i-wl:i] == word and dp[i-wl]:
                dp[i] = True
                break
    return dp[n]


def wb_tab(s, word_dict):
    """
    dp[i] = substring until index i can be split into words
    Time: O(n*w*l)
    Space: O(n) but no possible RecursionError
    """

    n = len(s)
    dp = [False] * (n+1)
    dp[0] = True
    for i in range(n+1):
        for word in word_dict:
            wl = len(word)
            if i >= wl and s[i-wl:i] == word and dp[i-wl]:
                dp[i] = True
                break
    return dp[n] 
    

if __name__ == "__main__":
    s1 = "catsandog"
    word_dict1 = ["cats","dog","sand","and","cat"]
    s2 = "leetcode"
    word_dict2 = ["leet", "code"]
    print("Memo:", wb_memo(s2, word_dict2))
    print("Tab:", wb_tab(s2, word_dict2))