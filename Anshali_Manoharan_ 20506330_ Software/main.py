import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from modules.image_processing import setup_image
from utils.node import Node
from modules import path_finding_cy_dynamic_training
from CONDENSATION import CONDENSATION_main
import matplotlib.pyplot as plt
from skimage.draw import line
from YOLO_inference import identify_object_from_contour, get_YOLO_model, evaluate_predicted_contours


# global Variables
clicked_point = None
img = None
scharr_x = None
scharr_y = None
edges = None
node_count = 0
nodes = None # stores nodes of the image
confirmed_seeds = []  # store confirmed seeds
final_outlines = []
seed_count = 0
hovered_point = None
total_frames = 0
thickness = 1 # thickness of hover path
final_evaluation_contours = [] # this will store the contour for input for CNN evaluation later.
training_buffer = np.empty((0, 2), dtype=np.int32) # for dynamic training
dynamic_training = False # decides whether dynamic training happens or not
dynamic_training_user = False

def set_up_snap_img(image_path):
    """
    Set up gradient and edge images for snapping the cursor - helps determine the seeds without exact precision by user.
    """
    global scharr_x, scharr_y, edges
    _, gray = setup_image(image_path)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    edges = cv2.magnitude(scharr_x, scharr_y)

def cursor_snap(seed_x, seed_y):
    """
    Snap the cursor to the point with the highest gradient magnitude.
    """
    global edges
    neighborhood_size = 7
    half_size = neighborhood_size // 2
    height, width = edges.shape
    max_magnitude = 0
    new_seed_x, new_seed_y = seed_x, seed_y

    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            neighbor_x = seed_x + i
            neighbor_y = seed_y + j
            if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                magnitude = edges[neighbor_y, neighbor_x]
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    new_seed_x, new_seed_y = neighbor_x, neighbor_y
    return new_seed_x, new_seed_y

def mouse_callback(event, x, y, flags, param):
    global clicked_point, hovered_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img.shape[1] and y < img.shape[0]:  # Check bounds
            snapped_x, snapped_y = cursor_snap(x, y)
            clicked_point = (snapped_x, snapped_y)
            print(f"Clicked and snapped to: {clicked_point}")
    if event == cv2.EVENT_MOUSEMOVE:
        if x < img.shape[1] and y < img.shape[0]:  # Check bounds
            hovered_point = (x, y)
        else:
            hovered_point = None


def update_nodes(seed_point):

    global nodes, seed_count, img, final_outlines, training_buffer, dynamic_training, dynamic_training_user


    # before taking new seed to perform expansion, append path from current spot to PREVIOUS SEED POINT
    if seed_count > 1: # only then a segment exists for dynamic training to happen
        if dynamic_training_user:
            dynamic_training = True
        else:
            dynamic_training = False
        if len(final_outlines[-1]) >= 32:
            training_buffer = np.vstack((training_buffer, final_outlines[-1][-32:]))
        else:
            if len(final_outlines[-1]) == 0:
                dynamic_training = False
            else:
                training_buffer = np.vstack((training_buffer, final_outlines[-1]))
    else:
        dynamic_training = False
    
    print("Updating nodes...")
    seed_y, seed_x = seed_point
    start = time.time()
    nodes = path_finding_cy_dynamic_training.live_wire(img.shape[0], img.shape[1], nodes, seed_x, seed_y, training_buffer, dynamic_training)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    # for the next time nodes expansion - all nodes need to be un-expanded.
    [setattr(node, 'expanded', False) for row in nodes for node in row]

def trace_path(hover_point, seed_count):
    # traces path back to only the LAST confirmed seed. This makes sure that the contour collected is SEQUENTIAL.
    global nodes
    if seed_count > 0:
        x, y = hover_point
        if x >= nodes[0].__len__() or y >= len(nodes) or x < 0 or y < 0:
            return []
        green_node = nodes[hover_point[1]][hover_point[0]]
        path = []
        while green_node != nodes[confirmed_seeds[-1][1]][confirmed_seeds[-1][0]]:
            path.append((green_node.location[1], green_node.location[0]))
            green_node = green_node.prevNode
            if green_node:
                continue
            else:
                return []
        return path
    else:
        return []

def draw_thick_path(image, gray, path, color, thickness):
    # draws path on the image.
    for x, y in path:
        cv2.circle(image, (x, y), thickness, color, -1)
        #image[y, x] = color

def IS_GUI():
    """
    Combines Intelligent Scissors with user interaction.
    input: image path, output: final object contour.
    """
    global clicked_point, img, confirmed_seeds, nodes, seed_count, hovered_point, total_frames, final_outlines, dynamic_training_user

    # File dialog to select image
    root = tk.Tk()
    root.withdraw()  # Hide root window

    file_path = filedialog.askopenfilename(
        title="Select a  Video File",
        filetypes=[("Video Files", "*.mp4"), ("Image Files", "*.jpg")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return None

    # extract number of frames from video
    if file_path.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)

        # check if video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
        else:
            # get total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # set up image and gradient data
    img, gray = setup_image(file_path)
    path_finding_cy_dynamic_training.initialise_link_costs(gray)  # sets up resources for link cost calculation
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # convert into bgr (to use later in segment mask)
    set_up_snap_img(file_path)  # image for cursor snap calculations
    nodes = np.array([[Node(row=x, column=y) for y in range(img.shape[1])] for x in range(img.shape[0])])

    # create window and set mouse callback
    cv2.namedWindow("Intelligent Scissors")
    cv2.setMouseCallback("Intelligent Scissors", mouse_callback)

    temp_img = img.copy()  # working copy of the image
    seed_count = 0  # stores if the user is pressing seed for the first time, else hover function should be activated

    # interaction loop
    while True:

        # Create a white panel for instructions
        instruction_width = 320
        instruction_height = img.shape[0]
        instruction_panel = np.ones((instruction_height, instruction_width, 3), dtype=np.uint8) * 255

        # Define instructions text
        instructions = [
            "Intelligent Scissors",
            "",
            "1. Click to add seed.",
            "2. Spacebar to confirm seed",
            "3. 'd' to toggle dynamic training",
            "4. ESC to exit"
        ]

        # add text for instruction panel
        y_start = 30
        for i, line in enumerate(instructions):
            cv2.putText(instruction_panel, line, (10, y_start + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # combine original image and instruction panel side-by-side
        display_img = np.hstack((temp_img.copy(), instruction_panel))


        # currently clicked point on screen with cursor snap
        if clicked_point:
            cv2.circle(display_img, clicked_point, 2, (0, 255, 0), -1)

        # highlight the current path
        if hovered_point and hovered_point[0] < img.shape[1] and hovered_point[1] < img.shape[0]:
            highlighted_path = trace_path(hovered_point, seed_count)
            draw_thick_path(display_img, gray, highlighted_path, (0, 0, 255), thickness)

        # re-plot confirmed seeds and traced paths to show user previous progress
        for seed in confirmed_seeds:
            cv2.circle(display_img, seed, 2, (0, 0, 255), -1)
        
        if seed_count >= 1:  # ONLY IF there are two or more seeds, there are paths to draw on the screen
            for path in final_outlines:
                draw_thick_path(display_img, gray ,path, (0, 0, 255), thickness)

        cv2.imshow("Intelligent Scissors", display_img)

        # HANDLING KEYBOARD INPUT
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == 100: # letter d (toggles dynamic training)
            if seed_count >= 1: # only then user can toggle dynamic training
                dynamic_training_user = not(dynamic_training_user)
                print("Dynamic Training set to: ", dynamic_training_user)
        elif key == 32:  # SPACEBAR to confirm seed
            seed_count += 1
            confirmed_seeds.append(clicked_point)
            cv2.circle(temp_img, clicked_point, 2, (0, 0, 255), -1)
            final_outlines.append(highlighted_path) # confirmed seed and preceding hover path stored
            update_nodes((clicked_point[1], clicked_point[0]))
        elif key == 13:  # ENTER  key used to confirm object outline
            print("Object outline complete.")  # all final paths confirmed by user stored in final_outlines (global variable)
            cv2.destroyAllWindows()
            return file_path
    cv2.destroyAllWindows() # add return None, None and print TERMINATED Process.


def get_line_coordinates(start, end):
    # forms the line segment.
    x1, y1 = start
    x2, y2 = end
    rr, cc = line(x1, y1, x2, y2)
    line_pixels = list(zip(rr, cc))
    return line_pixels

def connect_contours_with_lines(contours):
    # if there are gaps between the contour segments, this function adds simple line segments to ensure closed contour.
    closed_contour = []
    for i in range(len(contours)):
        closed_contour.append(contours[i])
        if i < len(contours) - 1:
            last_point = contours[i][-1]
            first_point = contours[i + 1][0]
            if last_point != first_point:
                closed_contour.append(get_line_coordinates(last_point, first_point))
    if closed_contour[0][0] != closed_contour[-1][-1]:
        closed_contour.append(get_line_coordinates(closed_contour[-1][-1], closed_contour[0][0]))
    return closed_contour

if __name__ == "__main__":  # actual main with IS and CONDENSATION

    print("Stage 1: Intelligent Scissors")
    original_file_path = IS_GUI()  # get user-selected image

    # cleaning up contour array (the coordinates are in sequential order.)
    # also reverse the order of each contour segment because intelligent scissors provides coordinates backwards.
    final_outlines = [np.flip(arr, axis=0).tolist() for arr in final_outlines if len(arr) > 0]
    final_outlines = connect_contours_with_lines(final_outlines)
    formatted_contours = [np.array(c, dtype=np.int32) for c in final_outlines]
    
    image = img
    flattened = np.array([item for sublist in formatted_contours for item in sublist])

    y_coords, x_coords = flattened[:,1], flattened[:,0]

    eval_status = identify_object_from_contour(img, x_coords, y_coords, get_YOLO_model())

    if eval_status !=None:
        print("\nObject identified by YOLOv11. Proceeding with evaluation.")
    else:
        print("\nObject not identified by YOLOv11. Proceeding without evaluation.")
    
    print("Stage 2: CONDENSATION Algorithm")
    
    final_contours_evaluation, frames = CONDENSATION_main(original_file_path, flattened, [])
    # final_contours_evaluation.shape => (number of frames, total number of coordinates in contour, 2)
    
    if eval_status != None:
        print("Stage 3: Evaluation with CNN Model")
        evaluation_data = zip(frames, final_contours_evaluation)
        evaluate_predicted_contours(get_YOLO_model(), evaluation_data, eval_status)


        
    
    

