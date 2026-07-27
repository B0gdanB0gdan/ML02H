"""
You are given a network of n nodes, labeled from 1 to n. 
You are also given times, a list of travel times as directed edges 
times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time
 it takes for a signal to travel from source to target.

We will send a signal from a given node k. 
Return the minimum time it takes for all the n nodes to receive the signal. 
If it is impossible for all the n nodes to receive the signal, return -1.

Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

Input: times = [[1,2,1]], n = 2, k = 1
Output: 1

Input: times = [[1,2,1]], n = 2, k = 2
Output: -1
"""

from collections import defaultdict
import heapq

def network_delay_time(times, n, k):

    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    heap = [(0, k)]
    visited = set()
    dist = {k: 0}
    dist[k] = 0

    while heap:

        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, w in graph[node]:
            if neighbor not in visited:
                new_dist = w + d
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

    if len(dist) < n:
        return -1

    return max(dist.values())

    
if __name__ == "__main__":
    times = [[2,1,1],[2,3,1],[3,4,1]]
    n = 4
    k = 2
    print(network_delay_time(times, n, k))