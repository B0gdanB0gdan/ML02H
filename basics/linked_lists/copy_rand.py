"""
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.
"""

from .core import ListNode, list_to_arr


class ListRandomNode(ListNode):
    def __init__(self, val: int, next = None, prev = None, rand = None):
        super().__init__(val, next, prev)
        self.rand = rand


def deepcopy_random_list(head: ListRandomNode) -> ListRandomNode:

    node_map = {}
    node = head
    copy_head = copy_tail = ListRandomNode(0)
    while node:
        copy_tail.next = ListRandomNode(node.val)
        node_map[node] = copy_tail.next
        node = node.next
        copy_tail = copy_tail.next
    copy_head = copy_head.next

    node1, node2 = copy_head, head
    while node2:
        node1.rand = None if not node2.rand else node_map[node2.rand]
        node2 = node2.next
        node1 = node1.next

    return copy_head


if __name__ == "__main__":

    node1 = ListRandomNode(7)
    node2 = ListRandomNode(13)
    node3 = ListRandomNode(11)
    node4 = ListRandomNode(10)
    node5 = ListRandomNode(1)

    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5

    node1.rand = None
    node2.rand = node1
    node3.rand = node5
    node4.rand = node3
    node5.rand = node1

    head = node1

    print(list_to_arr(deepcopy_random_list(head)))