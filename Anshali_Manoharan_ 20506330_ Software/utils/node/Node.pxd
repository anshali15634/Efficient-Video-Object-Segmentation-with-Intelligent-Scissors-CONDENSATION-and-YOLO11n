# utils/node/Node.pxd
cdef class Node:
    cdef public float cost
    cdef public tuple location
    cdef public Node prevNode
    cdef public bint expanded
