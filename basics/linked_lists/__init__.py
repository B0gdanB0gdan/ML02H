"""
LINKED LISTS -- a chain of nodes, each holding a value and a pointer to
the next node (and previous, for doubly-linked). Unlike arrays, there's
no random access -- you can only move forward (or backward, if doubly
linked) one node at a time. Almost every linked list problem is really
about POINTER MANIPULATION: carefully rewiring .next (and .prev) links
without losing track of nodes you still need.

The core question to ask first: "am I reversing/rewiring links, finding
a position relative to the end, merging multiple lists, or detecting a
cycle?" That routes you to one of the templates below.

=== TEMPLATE 1: Reversal (iterative, three-pointer) ===
Use when: reversing all or part of a list -- the single most common
linked-list manipulation.

    def reverse_list(head):
        prev = None
        curr = head
        while curr:
            next_node = curr.next     # save before overwriting
            curr.next = prev          # reverse the link
            prev = curr                # advance both pointers
            curr = next_node
        return prev                   # prev is the new head

Key idea: you need THREE pointers (prev, curr, next) because reversing
curr.next destroys the only way to reach the rest of the list -- next_node
must be saved BEFORE the rewrite happens.

=== TEMPLATE 2: Dummy head node ===
Use when: the head of the result list might change (e.g. merging,
removing nodes, including possibly removing the original head itself).
A dummy node avoids special-casing "is this the first node?" everywhere.

    def merge_two_lists(l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 if l1 else l2   # attach whatever's left
        return dummy.next               # skip the dummy itself

=== TEMPLATE 3: Fast/slow pointers ===
Use when: finding the middle, detecting a cycle, or finding the Nth node
from the end. (Full detail already covered in the Two Pointers cheat
sheet -- this is the same technique applied to linked lists specifically.)

    def middle_node(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow    # slow lands on the middle (2nd middle if even length)

=== TEMPLATE 4: Merge K lists (heap-based) ===
Use when: merging MORE than 2 sorted lists at once -- pairwise merging
would be O(n*k^2) in the worst case; a heap gets you O(n log k).

    import heapq

    def merge_k_lists(lists):
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))  # i breaks ties (nodes aren't comparable)

        dummy = ListNode(0)
        tail = dummy
        while heap:
            val, i, node = heapq.heappop(heap)
            tail.next = node
            tail = tail.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))

        return dummy.next

Common gotchas:
* Always save `curr.next` BEFORE overwriting it during reversal -- the
  #1 source of "lost the rest of the list" bugs.
* Dummy node: return `dummy.next`, NOT `dummy` itself -- easy to forget.
* Fast/slow: guard `fast and fast.next` (not just `fast`) before advancing
  fast by 2 -- crashes on odd-length lists otherwise.
* Merge K Lists: node objects aren't directly comparable in Python, so
  heap tuples need a tiebreaker (index i) BEFORE the node itself, or
  ties on `val` will crash trying to compare ListNode objects.
* When reversing a SUBLIST (not the whole list), you need to carefully
  track the nodes just before and after the sublist boundary to
  reconnect everything correctly afterward.
"""