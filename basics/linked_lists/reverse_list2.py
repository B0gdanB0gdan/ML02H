"""
Given the head of a singly linked list and two integers left and right where left <= right, 
reverse the nodes of the list from position left to position right, and return the reversed list.

Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Input: head = [5], left = 1, right = 1
Output: [5]
"""

from .core import ListNode, arr_to_list, list_to_arr


def reverse_between(head: ListNode, left: int, right: int):

    prev_left = None
    dummy = ListNode(val=0)
    dummy.next = head
    node_left = dummy
    for _ in range(left):
        prev_left = node_left
        node_left = node_left.next

    prev = None
    node_right = node_left
    for _ in range(right-left+1):
        next_node = node_right.next
        node_right.next = prev
        prev = node_right
        node_right = next_node

    prev_left.next = prev
    node_left.next = node_right
    return dummy.next


if __name__ == "__main__":
    print(list_to_arr(reverse_between(ListNode.from_arr([1,2,3,4,5]), left=2, right=4)))

