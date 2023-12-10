"""
Plot tree.
Plot decision boundaries.
"""
import networkx as nx
import matplotlib.pyplot as plt
from EoN import hierarchy_pos


def traverse_tree(tree):
    nodes, edges = [f"{tree.attribute}<={tree.threshold}"], []
    _traverse(tree, f"{tree.attribute}<={tree.threshold}", nodes, edges)
    return nodes, edges


def _traverse(tree, parent, nodes, edges):

    for subtree in tree.subtrees.values():
        if hasattr(subtree, "attribute"):

            nodes.append(f"{subtree.attribute}<={subtree.threshold}")
            edges.append((parent, f"{subtree.attribute}<={subtree.threshold}"))
            _traverse(subtree, f"{subtree.attribute}<={subtree.threshold}", nodes, edges)
        else:
            nodes.append(subtree)
            edges.append((parent, subtree))


def plot_tree(tree):

    G = nx.DiGraph()
    nodes, edges = traverse_tree(tree)
    G.add_nodes_from(nodes)

    G.add_edges_from(edges)
    T = nx.bfs_tree(G, source=f"{tree.attribute}<={tree.threshold}")
    pos = hierarchy_pos(T, root=f"{tree.attribute}<={tree.threshold}")
    nx.draw(T, pos=pos, with_labels=True, font_weight='bold')
    plt.show()
