from collections import Counter, defaultdict


def two_sum(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        comp = target - num
        if comp in seen:
            return [seen[comp], i]
        seen[num] = i
    return [-1, -1]


def counter_capabilities(arr: str):

    freq = Counter(arr)
    print(freq.most_common(4))
    print(freq[arr[0]])
    print(freq[arr[0]] == freq[arr[-1]])


def group_anagrams(strs):
    groups = defaultdict(list)

    for item in strs:
        key = tuple(sorted(item))
        groups[key].append(item)
    return list(groups.values())


if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], target=22))
    counter_capabilities("ana are mere")
    print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))