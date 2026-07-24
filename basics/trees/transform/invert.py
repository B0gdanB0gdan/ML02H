"""
Given the root of a binary tree, invert the tree, and return its root.
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import serialize, deserialize


def invert(root: TreeNode):
    if root is None:
        return None
    
    left = invert(root.right)
    right = invert(root.left)

    root.left = left
    root.right = right
    
    return root


if __name__ == "__main__":
    arr = [4,2,7,1,3,6,9]
    root = deserialize(arr)
    inverted = invert(root)
    print(serialize(inverted))