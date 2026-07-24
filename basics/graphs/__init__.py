"""
GRAPHS — a set of nodes (vertices) connected by edges, which may be directed
or undirected, weighted or unweighted, and MAY contain cycles.

Because cycles are possible, every graph traversal needs a "visited" set —
this is the single biggest thing that separates graphs from tree recursion.

When to use which traversal?
* BFS (queue): shortest path in an UNWEIGHTED graph, "minimum steps/levels"
  problems, multi-source spreading (e.g. Rotting Oranges).
* DFS (recursion or explicit stack): exploring all paths, detecting cycles,
  connected components, topological sort, backtracking-style path problems.
* Dijkstra (min-heap): shortest path in a WEIGHTED graph with non-negative
  weights only.
* Bellman-Ford: shortest path with possibly NEGATIVE weights, or need to
  detect negative cycles. Slower (O(V*E)) but more general than Dijkstra.
* Floyd-Warshall: need ALL-PAIRS shortest paths at once, small graph (V^3 ok).

The core question to ask first: "am I looking for reachability/existence,
shortest path, or an ordering (topological)?"
* Reachability / components → DFS or BFS, doesn't matter which
* Shortest path, unweighted  → BFS
* Shortest path, weighted    → Dijkstra (non-negative) or Bellman-Ford (can be negative)
* Ordering with dependencies → topological sort (DFS-based or Kahn's/BFS-based)
* "Are these connected / will adding this edge create a cycle?" → Union-Find

Union-Find (Disjoint Set) — use when the problem is about DYNAMICALLY
merging groups or checking connectivity, especially if edges are added
one at a time (e.g. Redundant Connection, Accounts Merge, Number of
Provinces). Path compression + union by rank → nearly O(1) per operation.

Cycle detection differs by graph type:
* Undirected graph: a visited neighbor that ISN'T the immediate parent
  means a cycle.
* Directed graph: need to track the current recursion path (a "gray" set,
  distinct from "fully visited/black"), since a directed graph can revisit
  a node through a different path without it being a cycle.

Topological sort — only valid on a Directed Acyclic Graph (DAG):
* DFS-based: postorder + reverse the finish order.
* Kahn's algorithm (BFS-based): repeatedly remove nodes with in-degree 0.
  Cycle exists if you can't process all nodes this way.

Common gotchas:
* Forgetting to mark a node visited BEFORE recursing into it (can cause
  infinite loops or exponential blowup on graphs with cycles, unlike trees).
* Directed vs undirected changes both cycle detection AND how you think
  about "connected" (strongly vs weakly connected).
* Multi-source BFS: start the queue with ALL sources at once (level 0),
  not one at a time — this is the trick behind Rotting Oranges, 01 Matrix.
* Grid problems ARE graphs: each cell is a node, edges connect adjacent
  cells — BFS/DFS/Union-Find all apply directly (Number of Islands, etc).

Almost every tree problem can be written as:

        answer_from_left = dfs(node.left)
        answer_from_right = dfs(node.right)

        combine(left_answer, right_answer)
"""