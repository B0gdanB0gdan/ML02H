"""
STACKS -- LIFO (last-in-first-out) structure. Two very different families
of problems live under this one category: (1) MATCHING/NESTING problems,
where a plain stack tracks "what's currently open, waiting to be closed,"
and (2) MONOTONIC STACK problems, where the stack is kept in strictly
increasing or decreasing order, used to efficiently find the nearest
greater/smaller element for every position in an array.

The core question to ask first: "am I matching/pairing up nested
structures (parentheses, tags), or am I looking for the nearest
greater/smaller element relative to each position in an array?" That
routes you to one of the two templates below -- they look quite different
in code, but are unified by the same core LIFO mechanic.

=== TEMPLATE 1: Matching / nesting (plain stack) ===
Use when: checking validity of nested/paired structures, or evaluating
expressions where operations need to happen in the reverse order they
were "opened."

    def valid_parentheses(s):
        pairs = {')': '(', ']': '[', '}': '{'}
        stack = []

        for char in s:
            if char in pairs:                      # it's a CLOSING bracket
                if not stack or stack[-1] != pairs[char]:
                    return False
                stack.pop()
            else:                                    # it's an OPENING bracket
                stack.append(char)

        return len(stack) == 0    # everything must have been closed

Key idea: every closing bracket must match whatever opening bracket was
MOST RECENTLY seen and not yet closed -- that "most recent, unclosed"
property is exactly what a stack (LIFO) naturally tracks.

Min Stack (track the minimum in O(1), alongside normal push/pop) --
push a (value, current_min_so_far) PAIR onto the stack, instead of just
the value:

    class MinStack:
        def __init__(self):
            self.stack = []      # each entry: (value, min_so_far_INCLUDING_this_value)

        def push(self, val):
            current_min = min(val, self.stack[-1][1]) if self.stack else val
            self.stack.append((val, current_min))

        def pop(self):
            self.stack.pop()

        def top(self):
            return self.stack[-1][0]

        def get_min(self):
            return self.stack[-1][1]

Key idea: storing the running min ALONGSIDE each value means popping
never "loses" the correct previous minimum -- it's baked into the entry
right below it.

=== TEMPLATE 2: Monotonic stack (nearest greater/smaller element) ===
Use when: for EVERY element, you need to know "where's the next element
that's bigger (or smaller) than me" -- brute force is O(n^2), monotonic
stack gets this to O(n).

    def daily_temperatures(temps):
        n = len(temps)
        result = [0] * n
        stack = []              # stores INDICES, kept in decreasing-temp order

        for i, temp in enumerate(temps):
            while stack and temps[stack[-1]] < temp:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index    # found the "next greater" for prev_index
            stack.append(i)

        return result

Key idea: the stack holds indices whose "next greater element" hasn't
been found YET. Every time a new element is bigger than what's on top of
the stack, that top element FINALLY has its answer -- pop it, record the
distance, and keep checking against the new top. Each index is pushed
once and popped at most once, giving O(n) total despite the nested-
looking while loop.

Common gotchas:
* Matching/nesting: always check `if not stack` before popping/peeking --
  a closing bracket with nothing open to match is an immediate invalid,
  not a crash waiting to happen.
* Matching/nesting: an empty stack at the very END is also required --
  leftover unclosed opening brackets means invalid too.
* Monotonic stack: decide up front whether you're storing VALUES or
  INDICES -- indices are usually more useful, since you can always look
  up the value (`temps[i]`) but can't recover the index from a bare value.
* Monotonic stack: the `while` loop popping condition determines
  increasing vs. decreasing stack, and greater vs. smaller search --
  getting the comparison direction backwards is the most common bug
  here; trace a short example (3-4 elements) by hand to confirm before
  trusting it.
* Despite the nested while-inside-for structure, monotonic stack
  solutions are O(n) overall, NOT O(n^2) -- each element is pushed once
  and popped at most once across the ENTIRE run, so total work across
  all iterations is bounded by 2n, not n^2. Worth stating this
  explicitly in an interview, since the nested loop looks worse than
  it actually is.
"""