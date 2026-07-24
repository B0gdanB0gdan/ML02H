"""
Given a binary tree, determine if it is height-balanced.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees 
of every node never differs by more than one.

So it can be seen that any value in a balanced binary tree can be searched in O(logN) time 
where N is the number of nodes in the tree. But if the tree is not height-balanced then in the worst case, 
a search operation can take O(N) time.
"""
from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize


def is_balanced_rec(root: TreeNode | None):
    if root is None:
        return 0


    height_left = is_balanced_rec(root.left)
    if height_left == -1:
        return -1
    height_right = is_balanced_rec(root.right)
    if height_right == -1:
        return -1
    if abs(height_left - height_right) > 1:
        return -1
    
    return 1 + max(height_left, height_right)


def is_balanced(root):
    return is_balanced_rec(root) > 0
    


if __name__ == "__main__":
    arr = [3,9,20,None,None,15,7] # level order serialization format
    root = deserialize(arr)
    print("Max depth:", is_balanced(root))