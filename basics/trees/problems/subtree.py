"""
Given the roots of two binary trees root and subRoot, 
return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. 
The tree tree could also be considered as a subtree of itself.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize
from collections import deque


def dfs_check(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    
    left = dfs_check(root1.left, root2.left)
    right = dfs_check(root1.right, root2.right)

    return left and right and root1.val == root2.val


def is_subtree(root: TreeNode, sub_root: TreeNode):
    """
    Time: O(n*m)
    Space: O(n+m)
    """
    queue = deque([root])

    while queue:

        node = queue.popleft()
        if node.val == sub_root.val and dfs_check(node, sub_root):
            return True

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return False


if __name__ == "__main__":
    root = deserialize([3,4,5,1,2])
    sub_root = deserialize([4,1,2])
    print(is_subtree(root, sub_root))

    root = deserialize([3,4,5,1,2,None,None,None,None,0])
    sub_root = deserialize([4,1,2])
    print(is_subtree(root, sub_root))