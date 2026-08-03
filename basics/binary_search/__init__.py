"""
BINARY SEARCH -- repeatedly halve a SEARCH SPACE to find a target or a
boundary in O(log n), instead of scanning linearly in O(n).

The core requirement: the search space must have some MONOTONIC property
-- i.e., a way to decide "go left half" or "go right half" that's
consistently correct, without needing to check every element.

The one question to ask first: "am I searching a literal sorted ARRAY for
a value/boundary, or am I searching an abstract ANSWER SPACE (a range of
possible numeric answers) for the smallest/largest value that satisfies
some condition?" The second case -- "binary search on the answer" -- is
the one people most often fail to recognize.

=== TEMPLATE 1: Classic binary search (find exact value) ===
Use when: array is sorted, looking for one specific value.

    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

=== TEMPLATE 2: Find first/last occurrence (boundary search) ===
Use when: array has duplicates, and you need the FIRST or LAST index
where target appears (not just "any" match).

    def find_first(arr, target):
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                right = mid - 1        # keep searching LEFT for an earlier match
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

    def find_last(arr, target):
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                left = mid + 1         # keep searching RIGHT for a later match
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

Key idea: don't return immediately on a match -- record it, then keep
narrowing in the direction that could reveal an even better (earlier/
later) match.

=== TEMPLATE 3: Search in rotated sorted array ===
Use when: array was sorted, then rotated at some unknown pivot.

    def search_rotated(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid

            if arr[left] <= arr[mid]:          # LEFT half is sorted
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:                                # RIGHT half is sorted
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

Key idea: exactly one half (left or right of mid) is ALWAYS properly
sorted, even after rotation -- figure out which half that is first
(`arr[left] <= arr[mid]`), then check if target falls within that
sorted half's range to decide which direction to go.

=== TEMPLATE 4: Binary search on the ANSWER (the one people miss) ===
Use when: the problem asks for a MINIMUM or MAXIMUM value satisfying
some condition, and you can write a function "is this candidate value
good enough?" that behaves monotonically (true for all values above/
below some threshold).

    def koko_eating_bananas(piles, h):
        def hours_needed(speed):
            return sum((pile + speed - 1) // speed for pile in piles)  # ceil division

        left, right = 1, max(piles)
        while left < right:
            mid = (left + right) // 2
            if hours_needed(mid) <= h:
                right = mid              # mid WORKS -- try to go slower (smaller)
            else:
                left = mid + 1             # mid too slow -- need to go faster

        return left

Key idea: you're not searching an array at all -- you're searching the
RANGE OF POSSIBLE ANSWERS (here, eating speeds 1 to max(piles)) for the
smallest value where a check function returns True. The recognition
trick: "minimize/maximize X such that condition(X) holds, and condition
gets easier to satisfy as X increases (or decreases)" -- that's binary
search on the answer, even though no array is being searched directly.

Common gotchas:
* `left <= right` vs `left < right` as the loop condition depends on
  whether you're searching for an exact index (use `<=`) or converging
  to a single boundary value (use `<`, template 4's style) -- mixing
  these up is the most common off-by-one source in this category.
* Find first/last: don't `return mid` immediately on a match -- you must
  keep narrowing past it to confirm it's truly the first/last one.
* Rotated array: always figure out WHICH HALF is sorted first (compare
  arr[left] vs arr[mid]) before deciding where target might be -- trying
  to reason about the unsorted half directly is what causes bugs here.
* Binary search on answer: the condition function must be MONOTONIC
  (true for a contiguous range of values, never "true, false, true" in
  an alternating pattern) -- if it's not monotonic, binary search on the
  answer is invalid and you need a different approach entirely.
"""