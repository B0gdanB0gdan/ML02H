"""
Given the root of a binary tree, flatten the tree into a "linked list":

The "linked list" should use the same TreeNode class where the right child pointer points 
to the next node in the list and the left child pointer is always null.
The "linked list" should be in the same order as a pre-order traversal of the binary tree.

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
"""


from basics.trees.core import TreeNode
from basics.trees.transform.serialize import serialize, deserialize


def flatten_rec(root, prev):
    if root is None:
        return prev
    
    prev = flatten_rec(root.right, prev)
    prev = flatten_rec(root.left, prev)
    
    root.right = prev
    root.left = None
    prev = root

    return prev


def flatten(root: TreeNode): 
    flatten_rec(root, None)
    return root
    
    
if __name__ == "__main__":
    arr = [1,2,5,3,4,None,6]
    root = deserialize(arr)
    inverted = flatten(root)
    print(serialize(inverted))