"""
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down 
to the farthest leaf node.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize


def max_depth(root: TreeNode | None):
    if root is None:
        return 0

    return 1 + max(
        max_depth(root.left), 
        max_depth(root.right)
    )


if __name__ == "__main__":
    arr = [3,9,20,None,None,15,7] # level order serialization format
    root = deserialize(arr)
    print("Max depth:", max_depth(root))