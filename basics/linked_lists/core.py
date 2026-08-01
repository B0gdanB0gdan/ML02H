class ListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

    @classmethod
    def from_arr(cls, arr: list):
        if not arr:
            return None
        head = cls(val=arr[0])
        tail = head
        for v in arr[1:]:
            tail.next = ListNode(val=v)
            tail = tail.next
        return head 

def list_to_arr(head: ListNode | None):
    node = head
    arr = []
    while node:
        arr.append(node.val)
        node = node.next
    return arr


def arr_to_list(arr: list = []):
    if not arr:
        return None
    head = ListNode(val=arr[0])
    tail = head
    for v in arr[1:]:
        tail.next = ListNode(val=v)
        tail = tail.next
    return head