"""
HASH MAPS — a data structure giving O(1) average-time lookup, insertion,
and deletion, keyed by any hashable value. The core capability this unlocks:
turning an O(n) "have I seen this before?" scan into an O(1) check.

The one question that signals "use a hashmap": "do I need to repeatedly
ask 'have I seen X before', 'how many times have I seen X', or 'what goes
together with X' -- across a single pass?" If yes, a hashmap turns an
O(n^2) brute force into O(n).

=== PATTERN 1: Complement / "have I seen this before" lookup ===
Use when: searching for a PAIR (or a specific counterpart) that satisfies
some condition -- the classic being "does some earlier element, combined
with this one, hit a target?"

Template:
    seen = {}   # value -> index (or just a set, if index isn't needed)
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

Why this beats sorting + two pointers: this preserves original indices
and works in a SINGLE pass without needing the array sorted at all --
two pointers would require sorting first (and lose original index info
unless you track it separately).

=== PATTERN 2: Frequency counting ===
Use when: you need to count occurrences of elements/characters, then
compare counts, find the most/least common, or check anagram-style
equality.

Template:
    from collections import Counter
    freq = Counter(arr)   # {value: count}

    # common operations:
    freq.most_common(k)          # top k by frequency
    freq[x] == freq[y]           # equality check between two multisets

=== PATTERN 3: Grouping by a derived key ===
Use when: you need to bucket items together based on some transformation
of each item (not the item itself) -- e.g., group words that are anagrams
of each other.

Template:
    from collections import defaultdict
    groups = defaultdict(list)
    for item in items:
        key = transform(item)     # e.g. tuple(sorted(item)) for anagrams
        groups[key].append(item)
    return list(groups.values())

The key insight: the SAME derived key means "these belong together" --
picking the right transform (sorted tuple, character count signature,
etc.) is the actual problem-solving step; the hashmap grouping itself
is mechanical once you have that.

=== PATTERN 4: Hashmap as a "seen" set for O(1) existence checks ===
Use when: you just need "does this exist / has this happened before,"
with no extra value attached -- use a `set`, not a full dict, since you
don't need to store anything beyond presence/absence.

    seen = set()
    for x in arr:
        if x in seen:
            # found a duplicate / repeat
            ...
        seen.add(x)

=== Where hashmaps show up INSIDE other patterns (not a new pattern,
    just worth naming explicitly since you'll see it constantly) ===
* Sliding window: tracking character counts within the current window
  (Longest Substring Without Repeating Characters, Minimum Window Substring)
* Prefix sums: mapping prefix-sum-value -> how many times seen (Subarray
  Sum Equals K)
* Graphs: adjacency lists ARE hashmaps (defaultdict(list)); visited
  tracking; Union-Find's underlying arrays are a specialized case of this
* Clone Graph / tree reconstruction: mapping original-node -> cloned-node

Common gotchas:
* Mutable types (lists) can't be dict keys/set elements -- convert to
  tuples first if you need to hash a sequence (e.g. tuple(sorted(word))
  for anagram grouping).
* Dictionary iteration order is insertion order in modern Python, but
  don't rely on this for correctness unless the problem guarantees it.
* `defaultdict(list)` / `defaultdict(int)` avoids repetitive
  "if key not in dict: dict[key] = ..." boilerplate -- use it by default
  for grouping/counting patterns.
"""