'''
## The Hungarian algorithm, also known as the Kuhn-Munkres algorithm, is a combinatorial optimization algorithm that solves the assignment problem in polynomial time. It finds the maximum matching in a weighted bipartite graph. The algorithm works in 𝑂(𝑉3)time, where 𝑉 is the number of vertices.

## Hungarian Algorithm Steps

### Subtract Row Minimums:

For each row of the cost matrix, find the smallest element and subtract it from every element in that row.

### Subtract Column Minimums:

For each column of the cost matrix, find the smallest element and subtract it from every element in that column.

### Cover Zeros with Minimum Lines:

Cover all zeros in the resulting matrix using a minimum number of horizontal and vertical lines.

### Create Additional Zeros:

If the minimum number of covering lines is equal to the number of rows (or columns), an optimal assignment exists among the zeros. If not, find the smallest element not covered by any line, subtract it from all uncovered elements, and add it to all elements covered by both horizontal and vertical lines. Repeat the process until the minimum number of covering lines equals the number of rows or columns.
### Assignment:

Assign tasks to workers by finding a zero in the matrix, making that assignment, and then covering the row and column of the assignment. Continue until all tasks are assigned.
'''

