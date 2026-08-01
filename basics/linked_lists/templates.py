from .core import ListNode, list_to_arr
import heapq


def reverse_list(head: ListNode):
    prev = None
    node = head
    while node:
        next = node.next
        node.next = prev
        prev = node
        node = next
    return prev


def merge_lists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1

    node1, node2 = l1, l2

    if node1.val <= node2.val:
        head = node1
        node1 = node1.next
    else:
        head = node2
        node2 = node2.next

    tail = head

    while node1 and node2:
        if node1.val <= node2.val:
            tail.next = node1
            node1 = node1.next
        else:
            tail.next = node2
            node2 = node2.next

        tail = tail.next

    tail.next = node1 if node1 else node2
    return head


def merge_lists_dummy_node(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1

    dummy = ListNode(val=0)
    node1, node2 = l1, l2
    tail = dummy
    while node1 and node2:
        if node1.val <= node2.val:
            tail.next = node1
            node1 = node1.next
        else:
            tail.next = node2
            node2 = node2.next
        tail = tail.next
    tail.next = node1 if node1 else node2
    return dummy.next


def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def merge_k_lists(lists):
    """
    merge(l1, l2) = res
    merge(l3, res = res2 ...
    O(nk)
    With min heap: O(n log k)
    """
    heap = []
    for i, head in enumerate(lists):
        heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(val=0)
    tail = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node))

    return dummy.next

if __name__ == "__main__":
    head1 = ListNode(
        val=0,
        next=ListNode(
            val=1,
            next=ListNode(
                val=2,
                next=ListNode(
                    val=3,
                    next=ListNode(val=4)
                )
            )
        )
    )
    print(list_to_arr(reverse_list(head1)))
    head1 = ListNode(
        val=1,
        next=ListNode(
            val=4,
            next=ListNode(
                val=5,
                next=ListNode(
                    val=6,
                )
            )
        )
    )
    head2 = ListNode(
        val=2,
        next=ListNode(
            val=3,
            next=ListNode(
                val=4,
                next=ListNode(
                    val=8,
                )
            )
        )
    )
    # print(list_to_arr(merge_lists(head1, head2)))
    print(list_to_arr(merge_lists_dummy_node(head1, head2)))
    print(middle_node(head1).val)
