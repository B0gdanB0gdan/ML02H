"""
Given the head of a linked list, remove the nth node from the end of the list and return its head.

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Input: head = [1], n = 1
Output: []

Input: head = [1,2], n = 1
Output: [1]
"""
from .core import ListNode, arr_to_list, list_to_arr


def remove_nth_from_end(head: ListNode, n: int):

    fast, slow = head, head
    for _ in range(n):
        fast = fast.next

    prev = None
    while fast:
        prev = slow
        slow = slow.next
        fast = fast.next

    prev.next = slow.next
    return head


if __name__ == "__main__":
    print(list_to_arr(arr_to_list([1,2,3,4,5])))
    print(list_to_arr(remove_nth_from_end(
        head = arr_to_list([1,2,3,4,5]),
        n = 2
    )))