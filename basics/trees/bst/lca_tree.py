"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def lca_tree(root: TreeNode, p: int, q: int):
    
    if root is None or root.val == p or root.val == q:
        return root
    left = lca_tree(root.left, p, q)
    right = lca_tree(root.right, p, q)

    if left and right:
        return root
    
    return left if left else right


if __name__ == "__main__":
    arr = [3,5,1,6,2,0,8,None,None,7,4]
    root = deserialize(arr)
    p = 4
    q = 1
    print(lca_tree(root, p, q).val)
