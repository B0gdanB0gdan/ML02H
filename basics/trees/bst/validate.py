"""
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys strictly less than the node's key.
The right subtree of a node contains only nodes with keys strictly greater than the node's key.
Both the left and right subtrees must also be binary search trees.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize


def validate(root, min_val=float('-inf'), max_val=float('inf')):
    if root is None:
        return True
    
    if not (min_val < root.val < max_val):
        return False
    
    return validate(root.left, min_val, root.val) and \
           validate(root.right, root.val, max_val)
    

if __name__ == "__main__":
    arr = [4,2,7,1,3]
    # Expected: [4,2,7,1,3,5]
    root = deserialize(arr)
    print(validate(root))
    arr = [5,1,4,None,None,3,6]
    root = deserialize(arr)
    print(validate(root))