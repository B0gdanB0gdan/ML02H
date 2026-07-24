"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def lca(root: TreeNode, p: int, q: int):
    if root is None:
        return None
    if p < root.val and q < root.val:
        return lca(root.left, p, q)
    elif p > root.val and q > root.val:
        return lca(root.right, p, q)
    else:
        return root.val
    
def lca_iter(root: TreeNode, p: int, q: int):
    while root:
        if p < root.val and q < root.val:
            root = root.left
        elif p > root.val and q > root.val:
            root = root.right
        else:
            return root.val
    return None

if __name__ == "__main__":
    arr = [3,5,1,6,2,0,8,None,None,7,4]
    root = deserialize(arr)
    p = 5
    q = 1
    print(lca(root, p, q))
    print(lca_iter(root, p, q))

