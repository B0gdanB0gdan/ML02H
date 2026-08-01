"""
SLIDING WINDOW -- maintain a CONTIGUOUS range [left, right] over an array
or string, expanding and shrinking it incrementally instead of re-scanning
every possible subarray from scratch (which would be O(n^2) or worse).

The core requirement for this pattern to be valid at all: the window's
"validity" must change MONOTONICALLY as you add/remove elements -- e.g.
adding a character can only ever make a window "more full" or "more
invalid," never both, depending on direction. If negative numbers or
non-monotonic effects are involved, sliding window is the WRONG tool --
think prefix sum + hashmap instead (see the Prefix Sums cheat sheet).

The one question to ask first: "is the window FIXED size (given directly
by the problem, e.g. 'subarray of length k') or VARIABLE size (grows/
shrinks based on a condition)?"

=== TEMPLATE 1: Fixed-size window ===
Use when: the problem gives you an explicit window size k.

    def max_average_subarray(nums, k):
        window_sum = sum(nums[:k])          # prime the first window
        best = window_sum

        for right in range(k, len(nums)):
            window_sum += nums[right] - nums[right - k]   # slide by one:
                                                             # add new, remove old
            best = max(best, window_sum)

        return best / k

Key idea: no `left` pointer tracking needed explicitly -- the window's
left edge is always `right - k + 1`, so you just subtract the element
falling out of range as you add the new one.

=== TEMPLATE 2: Variable-size window (expand right, shrink left) ===
Use when: window size isn't fixed -- you grow it until some condition is
violated, then shrink until it's valid again, tracking the best size seen.

    def longest_substring_without_repeat(s):
        seen = {}                # char -> most recent index
        left = 0
        best = 0

        for right, char in enumerate(s):
            if char in seen and seen[char] >= left:
                left = seen[char] + 1     # jump left PAST the duplicate
            seen[char] = right
            best = max(best, right - left + 1)

        return best

    def min_window_substring(s, t):
        from collections import Counter
        need = Counter(t)
        missing = len(t)          # total count still needed, across all chars
        left = 0
        best_len = float('inf')
        best_start = 0

        for right, char in enumerate(s):
            if need[char] > 0:
                missing -= 1
            need[char] -= 1

            while missing == 0:                     # window is currently valid
                if right - left + 1 < best_len:
                    best_len = right - left + 1
                    best_start = left
                need[s[left]] += 1
                if need[s[left]] > 0:
                    missing += 1                       # window about to become invalid
                left += 1

        return "" if best_len == float('inf') else s[best_start:best_start + best_len]

Generic variable-window skeleton, to adapt to new problems:

    def sliding_window_generic(arr, condition_fn):
        left = 0
        window_state = ...             # e.g. Counter(), running sum, set
        best = 0

        for right in range(len(arr)):
            # add arr[right] to window_state

            while not condition_fn(window_state):     # window invalid -> shrink
                # remove arr[left] from window_state
                left += 1

                # best win size
            best = max(best, right - left + 1)        # or min(), per the problem

        return best

Common gotchas:
* The shrink step MUST be a `while`, not an `if` -- a single shrink isn't
  always enough to restore validity (e.g. multiple duplicate chars).
* Fixed-size window: the subtraction index is `right - k`, not
  `right - k + 1` -- trace a k=2 or k=3 example by hand to get this right.
* Variable window "longest" problems (Longest Substring Without Repeat)
  update `best` UNCONDITIONALLY every iteration (window is always valid
  by construction, since you jump `left` forward instantly). "Shortest"
  problems (Minimum Window Substring) update `best` only INSIDE the
  `while valid` block, right before shrinking -- getting this placement
  backwards is a common bug.
* Character-count window state: use `Counter` or a plain dict, but be
  careful comparing "window contains at least X of each needed char" --
  a naive `all(count >= need[c])` check on every iteration is O(26) or
  O(distinct chars) extra work per step; the `missing` counter trick
  above avoids that by tracking a single running integer instead.
"""