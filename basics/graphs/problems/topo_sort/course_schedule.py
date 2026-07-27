"""
There are a total of num_courses courses you have to take, 
labeled from 0 to num_courses - 1. You are given an array prerequisites where 
prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
"0" depends on "1"
Return true if you can finish all courses. Otherwise, return false.

 
Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
"""

from collections import defaultdict, deque


def can_finish(num_courses: int, prerequisites: list[list[int]]):
    # [ai, bi] in the input list is actually and edge in other way around
    graph = defaultdict(list)
    for a, b in prerequisites:
        graph[b].append(a)

    incoming = [0] * num_courses
    for i in range(num_courses):
        for neighbor in graph[i]:
            incoming[neighbor] += 1

    queue = deque([course for course in range(num_courses) if incoming[course] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            incoming[neighbor] -= 1
            if incoming[neighbor] == 0:
                queue.append(neighbor)
    return len(order) == num_courses

if __name__ == "__main__":
    print(can_finish(num_courses=2, prerequisites=[[1,0]]))