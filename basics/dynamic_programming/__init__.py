"""
DP is a technique that solves a problem by splitting it into multiple overlapping subproblems.
The results of the subproblems are stored to be later used in order to avoid recomputation.

Reduces time complexity from exponential to polynomial.

When to use?
Property 1 - Optimal Substructure: optimal solution of the problem 
             is found by finding first optimal solutions of the subproblmes
Property 2 - Overlapping Subproblems: the result of the subproblmes are needed
             multiple times.

Approaches:
* Top-Down (Memoization): Recursion + Cache. Start with biggest subproblems first. 
                          Before recursion check if already precomputed.
* Bottom-Up (Tabulation): Iteration + Cache. Start with smallest subproblems first.
"""