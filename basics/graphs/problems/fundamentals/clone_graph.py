"""
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
"""


from ...core import GraphNode

def adjacency_list_to_graph_node(adj_list: list[list[int]]) -> GraphNode:
    def convert(adj_list, i, created):

        if i in created:
            return created[i]

        node = GraphNode(val=i+1)
        created[i] = node

        node.neighbors = [convert(adj_list, j-1, created) for j in adj_list[i]]

        return node

    return convert(adj_list, 0, {})


def clone_graph(node):
    """
    Time: O(V+E)
    Space: O(v+E)
    """
    def clone(node, created):

        if node.val in created:
            return created[node.val]

        copy_node = GraphNode(val=node.val)
        created[node.val] = copy_node
        copy_node.neighbors = [clone(neigh, created) for neigh in node.neighbors]
        return copy_node

    return clone(node, {})


if __name__ == "__main__":
    adj_list = [[2,4],[1,3],[2,4],[1,3]]
    node = adjacency_list_to_graph_node(adj_list)
    print(clone_graph(node))

