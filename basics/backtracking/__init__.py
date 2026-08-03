"""
BACKTRACKING -- systematically explore all possible choices by building a
partial solution incrementally, and ABANDONING ("backtracking") as soon as
you determine the current path can't lead to a valid answer. It's really
just DFS over a "decision tree," where each node represents a partial
solution and each edge represents one more choice made.

The universal skeleton, underlying every problem in this category:

    def backtrack(path, choices_remaining):
        if <path is a complete/valid solution>:
            record a COPY of path (path[:] -- not path itself, see gotchas)
            return                          # or don't return, if you want to keep exploring further

        for choice in choices_remaining:
            if <choice is invalid right now>:
                continue                      # skip (pruning)

            path.append(choice)               # CHOOSE
            backtrack(path, updated_remaining_choices)   # EXPLORE
            path.pop()                         # UN-CHOOSE (backtrack!)

The three-step rhythm -- CHOOSE, EXPLORE, UN-CHOOSE -- is the one thing
to internalize above all else. The "un-choose" step (undoing the choice
after the recursive call returns) is what makes this "backtracking"
rather than plain DFS/recursion -- it's how you correctly try the NEXT
sibling choice with a clean slate.

The core question to ask first: "am I choosing a SUBSET (include/exclude
each element), an ORDERING (arrange all elements), a COMBINATION that
sums/matches a target (may reuse elements or not), or placing things on a
GRID/BOARD subject to constraints?"

=== TEMPLATE 1: Subsets (include / exclude each element) ===
Use when: generate ALL possible subsets of a set.

    def subsets(nums):
        result = []

        def backtrack(start, path):
            result.append(path[:])              # every path is valid -- record it every time
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1, path)            # move PAST i -- no reuse, no earlier indices
                path.pop()

        backtrack(0, [])
        return result

Key idea: `start` prevents revisiting earlier elements or reusing the
same element twice -- each recursive call only considers elements AFTER
the current index, which is what generates each combination exactly once
without duplicates (given no duplicate values in input).

Subsets II (WITH duplicate values in input) -- sort first, then skip
adjacent duplicates AT THE SAME RECURSION DEPTH:

    def subsets_with_dup(nums):
        nums.sort()
        result = []

        def backtrack(start, path):
            result.append(path[:])
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i-1]:
                    continue                        # skip duplicate at this level
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()

        backtrack(0, [])
        return result

=== TEMPLATE 2: Permutations (ordering -- every element used exactly
    once, order matters) ===
Use when: generate all possible ORDERINGS of a full set.

    def permutations(nums):
        result = []

        def backtrack(path, used):
            if len(path) == len(nums):
                result.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack(path, used)
                path.pop()
                used[i] = False

        backtrack([], [False] * len(nums))
        return result

Key idea: unlike Subsets, you loop from index 0 EVERY time (not from a
"start" index) -- order matters, so you need to consider every unused
element as a candidate for the NEXT position, regardless of where it sits
in the original array. `used[]` tracks which elements are already placed.

=== TEMPLATE 3: Combination Sum (target-sum, MAY reuse elements) ===
Use when: find combinations that sum to a target, allowing repeated use
of the same element.

    def combination_sum(candidates, target):
        result = []
        candidates.sort()                    # enables early pruning

        def backtrack(start, path, remaining):
            if remaining == 0:
                result.append(path[:])
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break                       # sorted -- everything after is even bigger, stop early
                path.append(candidates[i])
                backtrack(i, path, remaining - candidates[i])   # i, NOT i+1 -- allows reuse
                path.pop()

        backtrack(0, [], target)
        return result

Key idea: passing `i` (not `i + 1`) to the recursive call is what allows
the SAME element to be picked again -- this is the key difference from
Subsets/Permutations, where you always move strictly forward.

=== TEMPLATE 4: Grid/board search (Word Search) ===
Use when: searching for a path/pattern through a 2D grid, where each
cell can only be used once PER PATH (but can be reused by a different
path attempt).

    def word_search(board, word):
        rows, cols = len(board), len(board[0])

        def backtrack(r, c, i):
            if i == len(word):
                return True                     # matched every character
            if (r < 0 or r >= rows or c < 0 or c >= cols
                    or board[r][c] != word[i]):
                return False

            temp = board[r][c]
            board[r][c] = '#'                    # mark visited IN PLACE (no extra set needed)

            found = (backtrack(r+1, c, i+1) or backtrack(r-1, c, i+1) or
                     backtrack(r, c+1, i+1) or backtrack(r, c-1, i+1))

            board[r][c] = temp                    # UN-CHOOSE -- restore for other paths
            return found

        for r in range(rows):
            for c in range(cols):
                if backtrack(r, c, 0):
                    return True
        return False

Key idea: mutating the grid IN PLACE (temporarily marking a cell as
visited, then restoring it) avoids needing a separate visited set --
directly mirrors the "choose, explore, un-choose" rhythm, just applied
to grid cells instead of a growing list.

Common gotchas:
* ALWAYS append `path[:]` (a copy), never `path` itself, when recording
  a result -- `path` keeps getting mutated by later backtracking, so
  storing a reference to it directly means every recorded "answer" ends
  up pointing to the same, final (usually empty) list.
* Forgetting the "un-choose" step (`path.pop()`, or restoring a grid
  cell) is the single most common backtracking bug -- without it, state
  leaks between sibling branches that should be independent.
* Subsets/Combination Sum: sorting first enables pruning (`break` once
  a candidate is too large) -- without sorting, you can't safely break
  early, only `continue`.
* Combination Sum uses `i` (reuse allowed) vs Subsets/Permutations use
  `i + 1` (no reuse) in the recursive call -- this single index detail
  is THE difference between "can reuse" and "can't reuse" problems.
* Word Search: remember to restore the temporarily-modified cell even
  when returning early / after an `or` short-circuits -- an easy spot to
  accidentally skip the cleanup step.
"""