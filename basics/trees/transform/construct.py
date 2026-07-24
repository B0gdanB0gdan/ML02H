"""
Given two integer arrays preorder and inorder 
where preorder is the preorder traversal of a binary tree 
and inorder is the inorder traversal of the same tree, 
construct and return the binary tree.

Example:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import serialize


def search(inorder, val, left, right):
    for i in range(left, right+1):
        if inorder[i] == val:
            return i
    return -1


def build_tree_rec(inorder, preorder, pre_idx, left, right):

    if left > right:
        return None
   
    in_idx = search(inorder, preorder[pre_idx[0]], left, right)
    if in_idx == -1:
        return None

    root = TreeNode(val=preorder[pre_idx[0]])
    pre_idx[0] += 1
    root.left = build_tree_rec(inorder, preorder, pre_idx, left, in_idx-1)
    root.right = build_tree_rec(inorder, preorder, pre_idx, in_idx+1, right)

    return root


def build_tree(inorder, preorder):
    left, right = 0, len(preorder)-1
    return build_tree_rec(inorder, preorder, [0], left, right)


if __name__ == "__main__":
    inorder = [9, 3, 15, 20, 7]
    preorder = [3, 9, 20, 15, 7]
    root = build_tree(inorder, preorder)
    print(serialize(root))