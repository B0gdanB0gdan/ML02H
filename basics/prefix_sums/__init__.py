"""
PREFIX SUMS — precompute cumulative sums so that "what's the sum of any
range [i, j]" becomes an O(1) lookup instead of an O(n) re-scan every time.

Core idea: build an array where prefix[i] = sum of all elements from
index 0 up to (but not including) index i. Then:

    sum(arr[i..j]) = prefix[j+1] - prefix[i]

Why this works: prefix[j+1] is "everything up to and including j."
prefix[i] is "everything up to but NOT including i." Subtracting removes
exactly the part before i, leaving exactly the range [i, j].

Template (building the prefix array):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i+1] = prefix[i] + arr[i]

    # range sum query, O(1):
    def range_sum(i, j):        # inclusive of both i and j
        return prefix[j+1] - prefix[i]

=== The key extension: prefix sum + hashmap ===
Use when: you need to count/find SUBARRAYS whose sum equals a target --
NOT just answer a single range-sum query.

Core insight: if prefix[j] - prefix[i] == target, that means the subarray
between i and j sums to target. Rearranged: prefix[i] == prefix[j] - target.
So as you scan and build up the running prefix sum, at each step you ask
"have I seen a prefix value equal to (current_prefix - target) before?"
-- and a hashmap gives you that answer in O(1).

Template (count subarrays summing to target):
    from collections import defaultdict
    counts = defaultdict(int)
    counts[0] = 1          # empty prefix (sum 0) has been "seen" once, by definition
    running_sum = 0
    result = 0

    for num in arr:
        running_sum += num
        result += counts[running_sum - target]   # how many earlier prefixes make this work
        counts[running_sum] += 1

    return result

Why counts[0] = 1 matters: this accounts for the case where a subarray
STARTING AT INDEX 0 itself sums exactly to target -- there's no "earlier"
prefix to subtract, so you need a placeholder representing "sum of zero
elements" to make the formula work uniformly.

=== Why this is NOT the same as sliding window ===
Sliding window requires the window's validity to move monotonically as you
grow/shrink it (e.g., all positive numbers -> sum only increases). The
moment NEGATIVE numbers are allowed, growing/shrinking the window can make
the sum go up AND down unpredictably -- sliding window's "shrink while
invalid" logic breaks down, because there's no guarantee shrinking fixes
anything. Prefix sum + hashmap sidesteps this entirely, since it doesn't
rely on any monotonic movement -- it just looks up "have I seen this exact
value before," which works regardless of positive/negative numbers.

The one question that tells you which to use: "can this array contain
negative numbers (or zero, in some framings)?" If yes, and the problem
is about subarray sums -- prefix sum + hashmap, not sliding window.

Common gotchas:
* Off-by-one on the prefix array's length (it's len(arr)+1, not len(arr)) --
  this is what makes the "j+1" vs "i" subtraction line up correctly.
* Forgetting counts[0] = 1 -- silently undercounts subarrays starting at
  index 0.
* 2D prefix sums (for matrix range-sum queries) follow the same idea, just
  with inclusion-exclusion across 4 corners instead of 2 endpoints.
"""