"""
Priority Queue = an abstract concept (an ADT — Abstract Data Type)
A priority queue is a concept: "a collection where you can insert elements, and always retrieve/remove the highest-priority one first" (highest could mean smallest or largest, depending on convention).
Heap = one specific, efficient implementation of that concept

A heap is a specialized tree-based data structure that keeps track of a minimum (or maximum)

A min-heap is a binary tree with one property: every parent node is smaller than 
(or equal to) its children. This is called the "heap property," and it's weaker 
than a BST's ordering - there's no left-vs-right rule, just "parent ≤ both children"
applied recursively down the whole tree.

O(1) peek at the minimum
O(log n) insert/remove

How it's actually stored — as an array, not a linked tree

In practice, heaps are usually implemented as a plain array:
parent of index i -> (i-1)//2
left child of i -> 2i+1
right child of i -> 2i+2

a heap is always a complete binary tree (filled left to right, no gaps)

2 3 4 5 7
0 1 2 3 4

parent of i=1: i=0
parent of i=2: i=0
children of i=1 (3): 2*i+1 (5) 2*i+2 (7)  


Python:
import heapq
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)

print(heap[0])          # 2 — peek at min, O(1)
print(heapq.heappop(heap))   # 2 — removes and returns min, O(log n)

heapq:
It's always a min-heap - smallest element pops first. If you need a max-heap, negate your values going in (heapq.heappush(heap, -val)) and negate again when popping.
You can push tuples, and it'll compare element-by-element — this is exactly why Dijkstra pushes (distance, node): it sorts primarily by distance, and node acts as a tiebreaker.
heapq.heapify(list) converts an existing list into a valid heap in-place, in O(n) — faster than pushing elements one at a time if you already have all the data upfront.

heapq is not a data structure itself — it's a set of functions that operate on a plain list
"""

import heapq

heap = []
heapq.heappush(heap, 5) # heapq function mutates the list to keep heap property
heapq.heappush(heap, 2) 
min_val = heapq.heappop(heap) # heapq function removes+returns the min, rearranges list
print(min_val)

