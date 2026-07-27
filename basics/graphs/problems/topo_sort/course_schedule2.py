"""
There are a total of num_courses courses you have to take, 
labeled from 0 to num_courses - 1. You are given an array prerequisites where 
prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return the ordering of courses you should take to finish all courses. 

If there are many valid answers, return any of them. 
If it is impossible to finish all courses, return an empty array.

Input: num_courses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
"""

from collections import deque, defaultdict


def course_schedule2(num_courses:int, prerequisites: list[list[int]]):

    graph = defaultdict(list)
    incoming = [0]*num_courses
    for a, b in prerequisites:
        graph[b].append(a) # b -> a
        incoming[a] += 1

    queue = deque([course for course in range(num_courses) if incoming[course] == 0])
    order = []
    while queue:
        course = queue.popleft()
        order.append(course)

        for neighbor in graph[course]:
            incoming[neighbor] -= 1 # one dependency is solved
            if incoming[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_courses else []


if __name__ == "__main__":
    print(course_schedule2(
        num_courses=4,
        prerequisites=[[1,0],[2,0],[3,1],[3,2]]
    ))