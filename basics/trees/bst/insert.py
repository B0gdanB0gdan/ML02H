"""
You are given the root node of a binary search tree (BST) 
and a value to insert into the tree. 
Return the root node of the BST after the insertion. 
It is guaranteed that the new value does not exist in the original BST.

Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. 
You can return any of them.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def insert(root, val):
    if root is None:
        return TreeNode(val)

    if val > root.val:
        root.right = insert(root.right, val)
    else:
        root.left = insert(root.left, val)
    return root


if __name__ == "__main__":
    arr = [4,2,7,1,3]
    val = 5
    # Expected: [4,2,7,1,3,5]
    root = deserialize(arr)
    new_root = insert(root, val)
    print(serialize(new_root))