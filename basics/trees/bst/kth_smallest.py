"""
Given the root of a binary search tree, and an integer k, 
return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize


def kth_smallest(root: TreeNode, k: int):
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return None
        
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right


if __name__ == "__main__":
    arr = [5,3,6,2,4,None,None,1]
    root = deserialize(arr)
    k = 3
    print(kth_smallest(root, k=k))