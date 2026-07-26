"""
dist from start to all nodes
only pos weights
Time: O(E log V)
Space: O(V)
"""
import heapq

def dijkstra(graph, start):
    """
    graph is adjacency list of key: [(neighbor, weight), ... ]
    
    - want the shortest distance to every other node
    - once you've found the shortest path to the currently-closest unvisited node, that distance is final and can never improve (greedy)
    
    If there was an edge with a negative weight: we lose O(ElogV) because nothing bounds how many times a node can be processed
    Infinite loops for cycles involving a negative edge


    Suppose 
    A --2--> B
    A --5--> C
    B --1--> C

    Why B is final when its popped?
    Suppose there were actually a shorter path to B that we haven't discovered yet: A -> X -> B but dist to B was 2
    but every unprocessed node has distance >= 2 because otherwise it would have been popped before B i.e. dist[X] >= 2
    dist to B through this path A -> X -> B = dist[X] + weight(X, B) >= 2+0

    """
    dist = {start: 0} # dist between start and all nodes
    # dist from start to start = 0 and from start to every other node is inf
    
    heap = [(0, start)]

    while heap:
        # pick closest node
        d, node = heapq.heappop(heap) # once we pop node its distance is finalized!
        # if the new distance (of a node we already found) is worse now ignore this path
        if d > dist.get(node, float('inf')):
            continue

        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist.get(neighbor, float('inf')): # relax neighbors
                # if new distance (prev inf) for the same neighbor is smaller update it
                # distance from start to this node is new dist
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return dist


if __name__ == "__main__":
    graph = {
        'A': [('B', 3), ('C', 1)],
        'B': [('A', 3), ('C', 1)],
        'C': [('A', 1), ('B', 1)]
    }
    # graph = {
    #         'A': [('B', 1)],
    #         'B': [('C', 1)],
    #         'C': [('B', -3)]
    #     }
    print(dijkstra(graph, 'A'))