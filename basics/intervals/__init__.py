"""
INTERVALS -- a list of (start, end) ranges. Almost every interval problem
is solved by SORTING first (by start time, almost always), then sweeping
left to right comparing each interval to whatever you've already processed.

The core question to ask first: "am I merging overlapping ranges, inserting
one new range into an existing sorted list, counting/removing overlaps, or
tracking how many ranges are simultaneously active at once?"

=== TEMPLATE 1: Merge overlapping intervals ===
Use when: given a list of intervals, combine every pair that overlaps.

    def merge_intervals(intervals):
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            last_end = merged[-1][1]
            if start <= last_end:              # overlaps with last kept interval
                merged[-1][1] = max(last_end, end)   # extend it
            else:
                merged.append([start, end])     # no overlap, start fresh

        return merged

Key idea: after sorting by start, you only ever need to compare the NEW
interval against the LAST one you've kept -- you never need to look
further back, because sorting guarantees nothing earlier could still be
"open" and unmerged by the time you reach a later interval.

=== TEMPLATE 2: Insert a new interval into an already-sorted, non-
    overlapping list ===
Use when: you're given a clean sorted list and ONE new interval to insert,
merging as needed.

    def insert_interval(intervals, new_interval):
        result = []
        i = 0
        n = len(intervals)

        # 1. add all intervals ending before new_interval starts
        while i < n and intervals[i][1] < new_interval[0]:
            result.append(intervals[i])
            i += 1

        # 2. merge all intervals that overlap with new_interval
        while i < n and intervals[i][0] <= new_interval[1]:
            new_interval[0] = min(new_interval[0], intervals[i][0])
            new_interval[1] = max(new_interval[1], intervals[i][1])
            i += 1
        result.append(new_interval)

        # 3. add whatever's left (starts after new_interval ends)
        while i < n:
            result.append(intervals[i])
            i += 1

        return result

=== TEMPLATE 3: Count max overlaps / min rooms needed at once ===
Use when: you need to know "what's the max number of intervals active
SIMULTANEOUSLY at any point" -- e.g. minimum meeting rooms.

    import heapq

    def min_meeting_rooms(intervals):
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        heap = []                      # tracks END times of ongoing meetings

        for start, end in intervals:
            if heap and heap[0] <= start:
                heapq.heappop(heap)      # a room freed up before this one starts
            heapq.heappush(heap, end)

        return len(heap)                # rooms still in use = answer

Key idea: the heap always holds the END times of currently-occupied rooms.
If the earliest-ending room finishes before (or exactly when) the next
meeting starts, that room can be REUSED -- pop it before pushing the new
end time. If the heap size never shrinks, you need a brand new room.

=== TEMPLATE 4: Greedy interval removal (non-overlapping intervals) ===
Use when: you need the MINIMUM number of intervals to remove so that
none of the remaining ones overlap.

    def erase_overlap_intervals(intervals):
        intervals.sort(key=lambda x: x[1])     # sort by END time, not start!
        count = 0
        prev_end = float('-inf')

        for start, end in intervals:
            if start >= prev_end:
                prev_end = end          # this interval is kept
            else:
                count += 1                # this interval overlaps -> remove it

        return count

Key idea: sorting by END time (not start) is the trick -- greedily keeping
whichever interval finishes EARLIEST leaves the most room for future
intervals to also fit without overlapping.

Common gotchas:
* Merge Intervals / Insert Interval: the overlap check boundary (`<=` vs
  `<`) depends on whether touching endpoints count as overlapping -- e.g.
  do [1,2] and [2,3] merge? Always clarify this before coding.
* Min Meeting Rooms: sort by START time (you're processing meetings in
  the order they begin), but the HEAP tracks end times -- don't mix these
  up.
* Erase Overlap Intervals: sort by END time, not start -- this is the
  one template in this category where sorting by start would give a
  wrong (or at least much harder to prove correct) answer.
* Always sort FIRST in every single one of these templates -- interval
  problems essentially never work correctly on unsorted input.
"""