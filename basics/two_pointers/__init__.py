"""
TWO POINTERS -- a family of techniques that replace an O(n^2) "check every
pair" scan with an O(n) or O(n log n) sweep, by exploiting sorted order or
a monotonic property of the array.

The core question to ask first: "are these pointers converging (opposite
ends, moving inward) or co-moving (same direction, different speeds)?"
That single question routes you to one of the two templates below.


=== TEMPLATE 1: Opposite-direction (converging) pointers ===
Use when: array is SORTED (or sortable), and you're looking for a
pair/triplet satisfying a sum/comparison condition. Requires sorted input
because the pointer-movement logic ("move left up" / "move right down")
only makes sense if you know moving a pointer strictly increases or
decreases the value it's looking at.

    def two_sum_sorted(arr, target):
        left, right = 0, len(arr) - 1
        while left < right:
            current = arr[left] + arr[right]
            if current == target:
                return [left, right]
            elif current < target:
                left += 1     # need a BIGGER sum -> move left pointer up
            else:
                right -= 1    # need a SMALLER sum -> move right pointer down
        return [-1, -1]

Extending to triplets (3Sum): sort the array, fix ONE element with an
outer loop, then run the two-pointer sweep on the remaining subarray for
each fixed element. O(n^2) total instead of O(n^3).

    def three_sum(nums):
        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue          # skip duplicate "fixed" elements
            left, right = i + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left-1]:
                        left += 1     # skip duplicate pairs
                    while left < right and nums[right] == nums[right+1]:
                        right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        return result

=== TEMPLATE 2: Same-direction (fast/slow) pointers ===
Use when: you need to partition/compact an array in place, or detect a
cycle, or find a position a fixed distance from another position -- both
pointers move in the SAME direction, usually at different speeds or with
a gap between them.

In-place compaction (e.g. remove duplicates from sorted array):

    def remove_duplicates(nums):
        slow = 0                        # slow = "next write position"
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
        return slow + 1                 # new length

Cycle detection (Floyd's algorithm):

    def has_cycle(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False

Nth-from-end (gap-based): advance one pointer n steps first, then move
both together -- the gap between them guarantees the right offset when
the leading pointer hits the end.

    def remove_nth_from_end(head, n):
        dummy = ListNode(0, head)
        fast = slow = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next

Common gotchas:
* Opposite-direction pointers REQUIRE sorted input -- if the array isn't
  sorted and you can't/shouldn't sort it (e.g. you need original indices,
  like Two Sum), use a hashmap instead, not two pointers.
* 3Sum's duplicate-skipping logic (the two inner `while` loops) is the
  single most common source of bugs in this pattern -- trace an input
  with duplicate values by hand before trusting it.
* Fast/slow pointers: always guard `fast and fast.next` (not just `fast`)
  before advancing fast by 2, or you'll crash on odd-length lists.
* For the "gap" technique (Nth from end), an off-by-one in how many steps
  to advance the lead pointer first is the most common bug -- trace a
  3-4 node list by hand to confirm the gap size.
"""