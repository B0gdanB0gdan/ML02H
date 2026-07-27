"""
There is a new alien language that uses the English alphabet, but the order of the letters is unknown.

You are given a list of strings words from the alien language's dictionary. 
It is claimed that the strings in words are sorted lexicographically by the rules of this new language.

If this claim is incorrect, and the given arrangement of strings in words cannot correspond to any order of letters, return "".

Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. 
If there are multiple solutions, return any of them.

A string a is lexicographically smaller than a string b if either of the following is true:

The first letter where they differ is smaller in a than in b.
a is a prefix of b and a.length < b.length.

Input: words = ["z","o"]
Output: "zo"

Input: words = ["hrn","hrf","er","enn","rfnn"]
Output: "hernf

Input: words = ["abc","ab"]
Output: ""
"""

from collections import defaultdict, deque


def foreignDictionary(words: list[str]) -> str:
    incoming = dict()
    for word in words:
        for c in word:
            incoming[c] = 0

    graph = defaultdict(list)
    for word1, word2 in zip(words[:-1], words[1:]):
        i = 0
        while i < len(word1) and i < len(word2) and word1[i] == word2[i]:
            i += 1

        if len(word1[i:]) == 0 or len(word2[i:]) == 0:
            if len(word1) > len(word2):
                return ""
            continue

        if word2[i] not in graph[word1[i]]:
            graph[word1[i]].append(word2[i])
            incoming[word2[i]] = incoming[word2[i]] + 1

    queue = deque([l for l in incoming if incoming[l] == 0])
    order = []
    while queue:
        l = queue.popleft()
        order.append(l)
        for neighbor in graph[l]:
            incoming[neighbor] -= 1
            if incoming[neighbor] == 0:
                queue.append(neighbor)

    return "" if len(order) != len(incoming) else "".join(order)



if __name__ == "__main__":
    print(
        foreignDictionary(
            words=["z","o"]
        )
    )
    print(
        foreignDictionary(
            words=["hrn","hrf","er","enn","rfnn"]
        )
    )
    print(
        foreignDictionary(
            words=["abc","ab"]
        )
    )