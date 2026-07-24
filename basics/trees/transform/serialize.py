"""
Serialize and Deserialize from BFS format
"""
from basics.trees.core import TreeNode
from collections import deque


def serialize(root):
    result = []
    queue = deque([root])

    while queue: # FIFO
         
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    while result[-1] is None:
        result.pop()

    return result


def deserialize(arr):
    
    if not arr:
        return None

    i = 0
    root = TreeNode(val=arr[i])
    queue = deque([root])

    while queue and i < len(arr):

        node = queue.popleft()

        i += 1
        if i < len(arr) and arr[i] is not None:
            node.left = TreeNode(val=arr[i])
            queue.append(node.left)
        i += 1
        if i < len(arr) and arr[i] is not None:
            node.right = TreeNode(val=arr[i])
            queue.append(node.right)
    
    return root


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(
            2,
            TreeNode(4),
            TreeNode(5)
        ),
        TreeNode(
            3,
            TreeNode(6),
            None
        )
    )
    
    arr = serialize(root)
    print("Serialize:", arr)

    arr = [3,9,20,None,None,15,7]
    root = deserialize(arr)
    print("Deserialize:", root)
    print("Serialize:", serialize(root))