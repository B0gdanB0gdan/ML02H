"""
Given the root of a binary tree, imagine yourself standing on the right side of it, 
return the values of the nodes you can see ordered from top to bottom.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize
from collections import deque


def right_side_view_rec(root: TreeNode):
    """
    Time: O(n)
    Space: O(h)
    bad if tree is deep: O(h) ~ O(n)
    """
    result = []
    def preorder_right_left(root: TreeNode, depth: int):
        if not root:
            return
        
        if depth == len(result):
            result.append(root.val)

        preorder_right_left(root.right, depth+1)
        preorder_right_left(root.left, depth+1)

    preorder_right_left(root, 0)
    return result 


def right_side_view(root: TreeNode):
    """
    Time: O(n)
    Space: O(n)

    bad if tree is wide: O(w) = max O(n/2) = O(n)
    """
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()

            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
    return result


if __name__ == "__main__":
    print(right_side_view(deserialize([1,2,3,4,None,None,None,5])))
    print(right_side_view_rec(deserialize([1,2,3,4,None,None,None,5])))