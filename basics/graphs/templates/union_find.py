"""
Dynamic connectivity
Used for undirected graphs
You have a bunch of items. 
Some become connected over time. 
At any moment, you want to know whether two items are in the same connected group.
Re-running BFS/DFS from scratch every time would be wasteful.
Union-Find solves this by maintaining groups as trees, where each group has one designated "representative" (the root), and checking "same group" just means checking "do these two nodes have the same root?"

Time: O(1)
Space: O(n)
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n)) # each node is its own parent
        self.rank = [0] * n # upper bound for height, matters for root mostly, has leftovers
    
    def find(self, x):
        # follow x's parent until you hit a node that is its own parent
        if self.parent[x] != x:
            # flatten the tree, point directly to last node
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y) # fast find because of shallow tree
        if root_x == root_y:
            return False

        # union by rank prevents the tree from becoming tall
        if self.rank[root_x] < self.rank[root_y]:
            # there are some nice properties for this
            # one the height/rank doesnt changed unless they have equal height/rank
            # two when merging two trees of same rank the number of nodes doubles but because
            # we can have max n node --> the tree will have height logn so even without compression
            # find will do worst case O(log n)
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


if __name__ == "__main__":
    
    uf = UnionFind(6) 
    print(uf.union(0, 1))
    print(uf.union(1, 2))
    print(uf.union(3, 4))

    print(uf.find(0))
    print(uf.find(1))
    print(uf.find(3))