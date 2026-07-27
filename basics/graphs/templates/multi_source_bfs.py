from collections import deque

def valid(i, j):
    pass


def multi_source_bfs(grid, sources):
    queue = deque(sources)      # seed the queue with ALL sources at once
    visited = set(sources)
    steps = 0

    while queue:
        for _ in range(len(queue)):   # process one full "wave" at a time
            r, c = queue.popleft()
            # process (r, c) — it's `steps` away from the nearest source
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if valid(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        steps += 1