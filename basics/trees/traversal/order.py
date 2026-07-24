from basics.trees.core import TreeNode
from collections import deque


def preorder(root):
    result = []
    def dfs(root):
        if not root:
            return
        
        result.append(root.val)
        dfs(root.left)
        dfs(root.right)

    dfs(root)
    return result


def preorder_iterative(root):
    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left) # last one is left
    
    return result
        

def postorder(root):
    result = []
    def dfs(root):
        if not root:
            return
        
        dfs(root.left)
        dfs(root.right)
        result.append(root.val)

    dfs(root)
    return result


def postorder_iterative(root):
    result = []
    stack = []

    while True:
        while root:
            stack.append(root)
            stack.append(root)
            root = root.left

        if not stack:
            return result

        root = stack.pop()
        if stack and root == stack[-1]:
            root = root.right
        else:
            result.append(root.val)
            root = None


def inorder(root):
    result = []
    def dfs(root):
        if not root:
            return
        
        dfs(root.left)
        result.append(root.val)
        dfs(root.right)

    dfs(root)
    return result


def inorder_iterative(root):
    result = []
    stack = []

    while True:
        while root:
            stack.append(root)
            root = root.left

        if not stack:
            return result
        
        root = stack.pop()
        result.append(root.val)

        root = root.right


def level_order(root):
    
    result = []
    queue = deque([root])

    while queue: # FIFO
        level_size = len(queue)
        level_vals = []
        for _ in range(level_size): 
            node = queue.popleft()
            level_vals.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level_vals)

    return result


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(
            2,
            TreeNode(4),
            TreeNode(5)
        ),
        TreeNode(
            3,
            TreeNode(6),
            None
        )
    )
    print("Preorder:", preorder(root))   # [1, 2, 4, 5, 3, 6]
    print("Preorder Iterative:", preorder_iterative(root))
    print("Inorder:", inorder(root))    # [4, 2, 5, 1, 6, 3]
    print("Inorder Iterative:", inorder_iterative(root))
    print("Postorder:", postorder(root))  # [4, 5, 2, 6, 3, 1]
    print("Postorder Iterative:", postorder_iterative(root))
    print("Level order:", level_order(root))  # [4, 5, 2, 6, 3, 1]