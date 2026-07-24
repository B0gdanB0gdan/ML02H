"""
Given the root of a binary tree, the value of a target node target, 
and an integer k, return an array of the values of all nodes that 
have a distance k from the target node.
You can return the answer in any order.
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
Output: [7,4,1]
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize
from collections import deque


def dist_k_nodes(root: TreeNode, target: int, k: int):
    """
    Time: O(n)
    Space: O(n)
    """
    def build_map(root: TreeNode, target: int, parent_map: dict):

        if not root:
            return None

        if root.val == target:
            return root

        left = build_map(root.left, target, parent_map)
        right = build_map(root.right, target, parent_map)

        parent_map[root.left] = root
        parent_map[root.right] = root

        if left:
            return left

        if right:
            return right

    def bfs(target_node, parent_map):

        queue = deque([target_node])
        result = []
        i = 0
        visited = {target_node}
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if i == k:
                    result.append(node.val)
                else:
                    if node.left and node.left not in visited:
                        queue.append(node.left)
                        visited.add(node.left)
                    if node.right and node.right not in visited:
                        queue.append(node.right)
                        visited.add(node.right)
                    if node in parent_map and parent_map[node] not in visited:
                        queue.append(parent_map[node])
                        visited.add(parent_map[node])
            if i == k:
                break
            i += 1
        return result
            
    parent_map = {}
    target_node = build_map(root, target, parent_map)
    return bfs(target_node, parent_map)


if __name__ == "__main__":
    print(dist_k_nodes(deserialize([3,5,1,6,2,0,8,None,None,7,4]), target=5, k=2))