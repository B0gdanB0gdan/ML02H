"""
TREES — a recursive data structure where each node has 0+ children,
and (for binary trees) at most 2: left and right.

Almost every tree problem is solved by asking:
"What do I need from my left and right subtrees to answer this node's question?"

Property 1 - Recursive Self-Similarity: a tree is defined in terms of itself
             (a tree's subtree is itself a valid, smaller tree).
Property 2 - No cycles: unlike graphs, there's exactly one path between
             any two nodes, so no visited-set is needed to avoid infinite loops.

When to use recursion vs iteration?
* Recursion (default): natural fit, since the problem IS defined recursively.
  Function signature usually returns whatever the parent needs from a child:
  a value, a boolean, a height, a pair (min, max), etc.
* Iteration: needed when asked explicitly, or for level-order traversal (BFS),
  or to avoid stack overflow on very deep/skewed trees.

Core traversal patterns (know all 4, recursive AND iterative):
* Preorder  (node, left, right)  — copy tree, serialize, prefix expressions
* Inorder   (left, node, right)  — BSTs give sorted order
* Postorder (left, right, node)  — delete tree, compute from children up
                                    (e.g. height, diameter, subtree sums)
* Level-order (BFS with a queue) — level-by-level problems, shortest path in
                                    unweighted tree, right/left side view

The one big decision to make before coding: "top-down" or "bottom-up"?
* Top-down: pass information FROM parent TO children as extra function
  arguments (e.g. current depth, running sum, allowed min/max for BST validity).
  Parent computes something, hands it down, doesn't wait for children's answer.
* Bottom-up: children compute their answer FIRST, then parent combines them
  (e.g. height = 1 + max(left_height, right_height); diameter passes through
  a "best so far" via nonlocal/return-tuple as children resolve first).
  This is just postorder traversal with a purpose.

Ask yourself: "does the answer for this node depend on information from
above (ancestors/depth) or information from below (children's results)?"
That answer alone tells you top-down vs bottom-up before you write any code.

BST-specific extra property to exploit:
* left subtree < node < right subtree  → lets you prune search space,
  turning O(n) search into O(log n) (if balanced), and makes inorder = sorted.

Common gotchas:
* Forgetting the null/None base case (return 0, None, True, or [] depending
  on what the function returns — get this wrong and everything above breaks).
* For "path sum" style problems: is a path allowed to bend through a node
  (visiting left AND right), or must it go strictly downward? Changes the
  recurrence significantly (see: Diameter / Max Path Sum vs Path Sum I/II).


1. Does each child solve an independent subproblem?
   -> Bottom-up DFS

2. Does the parent need to pass state downward?
   -> Top-down DFS

3. Am I processing levels?
   -> BFS

4. Is this exploiting BST ordering?
   -> Inorder / pruning

5. Does each node return ONE value or MULTIPLE values?
   -> tuple-return DP

6. Is the answer constrained to a root-to-leaf path,
   or can it bend through a node?
   -> Huge distinction (Path Sum vs Diameter/Max Path Sum)

7. Is the answer local to one subtree,
   or is it a global optimum?
   -> Often requires a nonlocal/global variable

8. What should dfs(node) return so that its parent
   has exactly the information it needs?  

   
1. Does solving it require answers from more than one subproblem before you can decide anything?
If yes → recursion. If you only ever need one path forward → iteration.

Validate BST: you need both left subtree valid and right subtree valid before you can answer for the current node → recursion.
BST LCA: value comparison alone tells you to go left or right, never both → iteration works fine.
General binary tree LCA (no ordering to exploit): you genuinely don't know which side p/q are on, so you must search both sides and combine → recursion.

2. Is there a point where you have to "come back" to a node after processing something further down (or further along)?
That's the hallmark of recursion — it's literally using the call stack as a to-do list of "come back here later." If the problem never needs to revisit a decision point, a loop replaces it exactly.

Post-order work (compute height, sum a subtree, validate) — needs to return upward → recursion.
A pointer just marching forward (search, insert, LCA-in-BST, two-pointer/sliding-window array problems) — never needs to come back → iteration.

3. Could the input make recursion arbitrarily deep?
Even when recursion is conceptually the natural fit, if the recursion depth scales with input size and could be large (e.g. a skewed tree, a long linked list, deep JSON), that's a real practical reason to prefer iteration or an explicit stack, since call-stack space is O(depth) and Python's default recursion limit (~1000) can bite you.

4. Quick tell: is the recursive call "tail position" — the last thing that happens, with nothing done to its return value?
If so, it can trivially become a loop, and usually should:
"""
