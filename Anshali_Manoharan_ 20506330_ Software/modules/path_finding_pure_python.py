import heapq
import numpy as np
from utils.node import Node
from modules.imgproc2 import setup_canny, setup_gradient_magnitude, setup_gradient_direction
import cv2
import math

wc = 0.43  # canny weight
wg = 0.43  # gradient magnitude weight
wd = 0.14  # gradient direction weight

# global variables for images
canny = None
final_gradient_magnitude = None
final_gradient_direction = None

# histograms
cost_canny = np.zeros(512, dtype=np.float32)
cost_grad = np.zeros(512, dtype=np.float32)

# 8-connectivity offsets
offset = [(-1, -1), (-1, 0), (-1, 1),
          (0, -1),          (0, 1),
          (1, -1), (1, 0),  (1, 1)]


def initialise_link_costs(gray: np.ndarray):
    global canny, final_gradient_magnitude, final_gradient_direction
    canny = setup_canny(gray).copy()
    final_gradient_magnitude = setup_gradient_magnitude(gray).copy()
    final_gradient_direction = setup_gradient_direction(gray).copy()


def compute_dynamic_cost_maps(training_buffer: np.ndarray):
    hist_canny = np.zeros(512, dtype=int)
    hist_grad = np.zeros(512, dtype=int)

    for i in range(training_buffer.shape[0]):
        x, y = training_buffer[i, 0], training_buffer[i, 1]
        hist_canny[min(int(canny[y, x] * 511), 511)] += 1
        hist_grad[min(int(final_gradient_magnitude[y, x] * 511), 511)] += 1

    max_c = max(hist_canny.max(), 1)
    max_g = max(hist_grad.max(), 1)

    for i in range(512):
        cost_canny[i] = 1.0 - (hist_canny[i] / float(max_c))
        cost_grad[i] = 1.0 - (hist_grad[i] / float(max_g))


def is_valid_coordinates(x: int, y: int, height: int, width: int) -> bool:
    return 0 <= x < height and 0 <= y < width


def compute_link_cost(px: int, py: int, qx: int, qy: int, dynamic_training: bool) -> float:
    fc = canny[qx, qy]
    fg = final_gradient_magnitude[qx, qy]
    dir_p = final_gradient_direction[px, py]
    dir_q = final_gradient_direction[qx, qy]
    fd = abs(dir_p - dir_q)
    fd = min(fd, 1.0 - fd)  # Cyclic wrap

    if not dynamic_training:
        return wc * fc + wd * fd + wg * fg

    fc = cost_canny[min(int(fc * 511), 511)]
    fg = cost_grad[min(int(fg * 511), 511)]
    return wc * fc + wd * fd + wg * fg


def live_wire(height: int, width: int, nodes: np.ndarray, seed_x: int, seed_y: int,
              training_buffer: np.ndarray, dynamic_training: bool) -> np.ndarray:
    if dynamic_training:
        compute_dynamic_cost_maps(training_buffer)

    max_link_cost = 4 * width * height - 3 * (width + height) + 2
    seed_node: Node = nodes[seed_y][seed_x]
    seed_node.cost = 0
    seed_node.prevNode = None

    buckets = [[] for _ in range(max_link_cost + 1)]
    heapq.heappush(buckets[0], seed_node)
    expanded_count = 0

    for bucket in buckets:
        if not bucket:
            continue
        while bucket:
            current_node = heapq.heappop(bucket)
            if not current_node.expanded:
                x, y = current_node.location
                current_node.expanded = True
                expanded_count += 1

                for dx, dy in offset:
                    nx, ny = x + dx, y + dy
                    if is_valid_coordinates(nx, ny, height, width):
                        neighbour_node = nodes[nx][ny]
                        new_cost = current_node.cost + compute_link_cost(x, y, nx, ny, dynamic_training)
                        if new_cost < neighbour_node.cost:
                            neighbour_node.cost = new_cost
                            neighbour_node.prevNode = current_node
                            heapq.heappush(buckets[int(new_cost)], neighbour_node)

    print("Number of nodes expanded:", expanded_count)
    return nodes
