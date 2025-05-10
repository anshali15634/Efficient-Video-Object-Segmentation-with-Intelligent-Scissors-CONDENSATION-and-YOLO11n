# Modules and Imports
import heapq
import numpy as np
cimport numpy as np
from utils.node import Node
from utils.node cimport Node
from modules.image_processing import setup_canny, setup_gradient_magnitude, setup_gradient_direction
from libc.math cimport acos, sqrt, pi, fabs, exp
from libc.stdlib cimport malloc, free
import cv2

# constants
cdef double wc = 0.4  # canny weight
cdef double wg = 0.3  # gradient magnitude weight
cdef double wd = 0.3  # gradient direction weight

# global variables for images
cdef np.float32_t[:, :] canny
cdef np.float32_t[:, :] final_gradient_magnitude
cdef np.float32_t[:, :] final_gradient_direction

# histograms
cdef float cost_canny[512]
cdef float cost_grad[512]

cdef list offset = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),         (0, 1),
                    (1, -1), (1, 0), (1, 1)]

def initialise_link_costs(np.ndarray[float, ndim=2] gray):  # Precomputes edge detection and gradients
    global canny, final_gradient_magnitude, final_gradient_direction
    cdef int height = gray.shape[0]
    cdef int width = gray.shape[1]
    canny = setup_canny(gray).copy()
    final_gradient_magnitude = setup_gradient_magnitude(gray).copy()
    final_gradient_direction = setup_gradient_direction(gray).copy()

cdef void compute_dynamic_cost_maps(int[:, :] training_buffer):

    cdef int hist_canny[512], hist_grad[512]
    cdef int i, x, y
    cdef int max_c = 1, max_g = 1

    for i in range(512):
        hist_canny[i] = 0
        hist_grad[i] = 0

    for i in range(training_buffer.shape[0]):
        x = training_buffer[i, 0]
        y = training_buffer[i, 1]

        # scale float to 0-511 before casting to int for binning
        hist_canny[min(int(canny[y, x] * 511), 511)] += 1
        hist_grad[min(int(final_gradient_magnitude[y, x] * 511), 511)] += 1

    for i in range(512):
        if hist_canny[i] > max_c:
            max_c = hist_canny[i]
        if hist_grad[i] > max_g:
            max_g = hist_grad[i]

    for i in range(512):
        cost_canny[i] = 1.0 - (hist_canny[i] / <float>max_c)
        cost_grad[i]  = 1.0 - (hist_grad[i] / <float>max_g)

def is_valid_coordinates(int x, int y, int height, int width):  # Check if coordinates are valid
    return 0 <= x < height and 0 <= y < width

cdef double compute_link_cost(int px, int py, int qx, int qy, bint dynamic_training):  # Total link cost

    cdef float fc = canny[qx, qy]
    cdef float fg = final_gradient_magnitude[qx, qy]
    cdef float dir_p = final_gradient_direction[px, py]
    cdef float dir_q = final_gradient_direction[qx, qy]
    cdef float fd = fabs(dir_p - dir_q)
    fd = min(fd, 1.0 - fd)  # cyclic wrap
    if not dynamic_training:
        return wc * fc + wd * fd + wg * fg # in range 0-1
    
    # else we have dynamic training component to compute!
    fc = cost_canny[min(int(fc * 511), 511)]
    fg = cost_grad[min(int(fg * 511), 511)]
    # fd remains the same (static feature)
    return wc * fc + wd * fd + wg * fg

cpdef np.ndarray[object, ndim=2] live_wire(int height, int width, np.ndarray[object, ndim=2] nodes, int seed_x, int seed_y, int[:,:] training_buffer, bint dynamic_training):
    """
    Compute live-wire segmentation using dynamic programming and bucket-sorting.
    """
    if dynamic_training:
        compute_dynamic_cost_maps(training_buffer)
    cdef int max_link_cost = 4 * width * height - 3 * (width + height) + 2 # assuming max cost between one node is 1
    cdef Node seed_node = nodes[seed_y][seed_x]
    seed_node.cost = 0
    seed_node.prevNode = None
    cdef list buckets = [[] for _ in range(max_link_cost + 1)]
    heapq.heappush(buckets[0], seed_node)
    cdef int expanded_count = 0

    cdef Node current_node, neighbour_node
    cdef int x, y, dx, dy
    cdef double newCost

    for bucket in buckets:
        if not bucket:
            continue
        while len(bucket) != 0:
            current_node = heapq.heappop(bucket)
            if not current_node.expanded:
                x, y = current_node.location
                current_node.expanded = True
                expanded_count += 1
                for dx, dy in offset:
                    if is_valid_coordinates(x + dx, y + dy, height, width):
                        neighbour_node = nodes[x + dx][y + dy]
                        newCost = current_node.cost + compute_link_cost(x, y, x + dx, y + dy, dynamic_training)
                        if newCost < neighbour_node.cost:
                            neighbour_node.cost = newCost
                            neighbour_node.prevNode = current_node
                            heapq.heappush(buckets[int(newCost)], neighbour_node)
    print("Number of nodes expanded:", expanded_count)
    return nodes
