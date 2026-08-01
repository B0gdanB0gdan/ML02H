"""
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
"""


from .core import ListNode, list_to_arr


def reorder_list(head: ListNode):

    # find middle point
    slow = fast = head
    prev_to_slow = None
    while fast and fast.next:
        prev_to_slow = slow
        slow = slow.next
        fast = fast.next.next

    prev_to_slow.next = None
    # reverse second half
    prev = None
    node = slow
    while node:
        next = node.next
        node.next = prev
        prev = node
        node = next

    # # merge
    node1, node2 = head, prev
    dummy = ListNode(val=0)
    tail = dummy
    while node1 and node2:
        tail.next = node1
        tail = tail.next
        node1 = node1.next
        tail.next = node2
        tail = tail.next
        node2 = node2.next

    tail.next = node1 if node1 else node2
    return dummy.next
        


if __name__ == "__main__":
    print(list_to_arr(reorder_list(ListNode.from_arr([1, 2, 3, 4, 5, 6, 7, 8]))))