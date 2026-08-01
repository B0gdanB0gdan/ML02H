
from collections import defaultdict


def build_prefix(arr):
    n = len(arr)
    prefix = [0] * (n + 1) # prefix[j] excludes j
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    return prefix


def range_sum(prefix, i, j):
    return prefix[j+1] - prefix[i]


def count_subarrays_target(nums, target):
    counts = defaultdict(int)
    counts[0] = 1
    running_sum = 0
    result = 0
    for num in nums:
        running_sum += num
        result += counts[running_sum - target]
        # running_sum - target = prefix_j - target = prefix_i
        # if prefix_i exists
        counts[running_sum] += 1 # negative numbers allowed
    return result


if __name__ == "__main__":
    prefix = build_prefix([1,2,3,4,5])
    print(range_sum(prefix, 1, 3))
    print(count_subarrays_target([1,2,3,1,2,4,2,1], target=3))