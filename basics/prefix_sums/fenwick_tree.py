class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n+1) # 1-indexed
        # each index holds the prefix sum of an interval ending at that index

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        """prefix sum until i"""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def range_query(self, left, right):
        """sum of values in interval left <= i <= right """
        return self.query(right) - self.query(left-1)


if __name__ == "__main__":
    arr = [1,3,5]
    tree = FenwickTree(len(arr))
    for i in range(len(arr)):
        tree.update(i+1, arr[i])

    print(tree.query(3))
    