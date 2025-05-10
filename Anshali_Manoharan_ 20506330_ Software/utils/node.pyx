# utils/node.pyx
cimport cython
from libc.math cimport INFINITY

cdef class Node:
    def __cinit__(self, int row, int column):
        self.cost = float(INFINITY)
        self.location = (row, column)
        self.prevNode = None
        self.expanded = False
 
    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.cost < other.cost
