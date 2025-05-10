'''
This version of IS is for testing Intelligent Scissor's accuracy against the PASCAL VOC 2012 dataset.
'''

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from modules.image_processing import setup_image
from utils.node import Node
from modules import path_finding_cy_dynamic_training
import os
import pandas as pd
from image_similarity_measures.quality_metrics import fsim, ssim


# Global Variables
clicked_point = None
img = None
scharr_x = None
scharr_y = None
edges = None
node_count = 0
nodes = None # stores nodes of the image
confirmed_seeds = []  # Store confirmed seeds
final_outlines = []
seed_count = 0
hovered_point = None
total_frames = 0
thickness = 2
training_buffer = np.empty((0, 2), dtype=np.int32) # for dynamic training
dynamic_training = False # decides whether dynamic training happens or not
dynamic_training_user = False # user's preference to turn on or off

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
    """
    Handle mouse events -> takes (x, y) coordinates and returns the snapped coordinates for display to user.
    """
    global clicked_point, hovered_point
    if event == cv2.EVENT_LBUTTONDOWN:
        snapped_x, snapped_y = cursor_snap(x, y)
        clicked_point = (snapped_x, snapped_y)
        print(f"Clicked and snapped to: {clicked_point}")
    if event == cv2.EVENT_MOUSEMOVE:
        hovered_point = (x, y)  # update hovered coordinates


def update_nodes(seed_point):
    global nodes, seed_count, img, final_outlines, training_buffer, dynamic_training_user, dynamic_training

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
    nodes = path_finding_cy_dynamic_training.live_wire(img.shape[0], img.shape[1], nodes, seed_x, seed_y, training_buffer, dynamic_training)
    # for the next time nodes expansion - all nodes need to be un-expanded.
    [setattr(node, 'expanded', False) for row in nodes for node in row]

def trace_path(hover_point, seed_count):
    # traces path back to only the LAST confirmed seed. This makes sure that the contour collected is SEQUENTIAL.
    global nodes
    if seed_count > 0:
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


def mse(image1, image2):
    # ensure the images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    return np.mean((image1 - image2) ** 2)

def dice_coefficient(mask1, mask2):
    """
    Compute the Dice coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 1.0  # If both masks are empty
    return (2.0 * intersection) / sum_masks

def iou_score(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0  # If both masks are empty
    return intersection / union

def evaluation(user_img, eval_img_path, spreadsheet_path):
    eval_img = cv2.imread(eval_img_path)
    eval_img = cv2.cvtColor(eval_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
    target_rgb = [224, 224, 192]  # target color for annotation
    contour_rgb = [0, 255, 0]     # contour color

    annotation_mask = np.all(eval_img == target_rgb, axis=-1).astype(np.uint8)
    user_mask = np.all(user_img == contour_rgb, axis=-1).astype(np.uint8)

    desktop_path = os.path.expanduser("~/Desktop/")
    user_mask_path = os.path.join(desktop_path, "user_mask.png")
    annotation_mask_path = os.path.join(desktop_path, "annotation_mask.png")

    cv2.imwrite(user_mask_path, user_mask * 255)
    cv2.imwrite(annotation_mask_path, annotation_mask * 255)

    # Compute Dice and IoU
    dice_score = dice_coefficient(user_mask, annotation_mask)
    iou = iou_score(user_mask, annotation_mask)

    # Convert to 3-channel for FSIM/SSIM
    user_mask_rgb = np.stack((user_mask,) * 3, axis=-1)
    annotation_mask_rgb = np.stack((annotation_mask,) * 3, axis=-1)

    fsim_score = fsim(user_mask_rgb, annotation_mask_rgb)
    ssim_score = ssim(user_mask_rgb, annotation_mask_rgb)
    mse_score = mse(user_mask_rgb, annotation_mask_rgb)

    image_id = os.path.splitext(os.path.basename(eval_img_path))[0]

    headers = ["Image ID", "FSIM", "SSIM", "MSE", "Dice", "IoU"]
    row_data = [image_id, fsim_score, ssim_score, mse_score, dice_score, iou]

    try:
        if os.path.exists(spreadsheet_path):
            df = pd.read_csv(spreadsheet_path)
        else:
            df = pd.DataFrame(columns=headers)

        new_row = pd.DataFrame([row_data], columns=headers)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(spreadsheet_path, index=False)
        print(f"Metrics for {image_id} appended to {spreadsheet_path}")
    except Exception as e:
        print(f"Error writing to spreadsheet: {e}")


def draw_thick_path(image, path, color, thickness):

    for x, y in path:
        cv2.circle(image, (x, y), thickness, color, -1)

def IS_GUI():

    global clicked_point, img, confirmed_seeds, nodes, seed_count, hovered_point, total_frames, dynamic_training_user, final_outlines

    # file dialog to select image
    root = tk.Tk()
    root.withdraw()  # hide root window

    file_path = filedialog.askopenfilename(
        title="Select an Image or Video File",
        filetypes=[("Image and Video Files", "*.jpg *.png *.jpeg *.bmp *.mp4")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    # extract number of frames from video
    if file_path.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
        else:
            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up image and gradient data
    img, gray = setup_image(file_path)
    path_finding_cy_dynamic_training.initialise_link_costs(gray)  # Sets up resources for link cost calculation
    set_up_snap_img(file_path)  # Sets up image for cursor snap calculations
    nodes = np.array([[Node(row=x, column=y) for y in range(img.shape[1])] for x in range(img.shape[0])])

    # Create window and set mouse callback
    cv2.namedWindow("Image Viewer")
    cv2.setMouseCallback("Image Viewer", mouse_callback)

    print("\nOpening Image Viewer...")
    print("(1) Click ESC to exit.")
    print("(2) Click on the image to add a seed.")
    print("(3) Press the spacebar to confirm your seed and start execution.")

    temp_img = img.copy()  # Create a working copy of the image
    seed_count = 0  # stores whether the user is pressing seed for the first time, else hover function should be activated

    while True:
        display_img = temp_img.copy()

        # Display currently clicked point on screen with cursor snap
        if clicked_point:
            cv2.circle(display_img, clicked_point, 2, (0, 255, 0), -1)

        # Highlight the current path
        highlighted_path = trace_path(hovered_point, seed_count)
        draw_thick_path(display_img, highlighted_path, (0, 255, 0), thickness)

        # Re-plot confirmed seeds and traced paths to show user previous progress
        for seed in confirmed_seeds:
            cv2.circle(display_img, seed, 2, (0, 0, 255), -1)
        if seed_count >= 1:  # Only if there are two or more seeds, there are paths to draw on the screen
            for path in final_outlines:
                draw_thick_path(display_img, path, (0, 255, 0), thickness)

        cv2.imshow("Image Viewer", display_img)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == 100: # letter d (toggles dynamic training)
            dynamic_training_user = not(dynamic_training_user)
            print("Dynamic Training set to: ", dynamic_training_user)
        elif key == 32:  # Space bar to confirm seed
            seed_count += 1
            confirmed_seeds.append(clicked_point)  # Add seed to confirmed list
            cv2.circle(temp_img, clicked_point, 2, (0, 0, 255), -1)  # Mark on temp_img
            final_outlines.append(highlighted_path)
            update_nodes((clicked_point[1], clicked_point[0]))
        elif key == 13:  # Enter key to confirm object outline
            print("Object outline complete.")  # All final paths confirmed by user stored in final_outlines
            return file_path, display_img
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    original_file_path, contour_img = IS_GUI() # get user selected image from dataset too
    image_id = os.path.splitext(os.path.basename(original_file_path))[0] # dataset image id from path
    eval_img_path = r"C:\\Users\\Manoharan\\Desktop\\Annotations\\"+image_id+".png"
    spreadsheet_path = r"C:\\Users\\Manoharan\\Desktop\\IS_evaluation.csv"
    # final_outlines = [arr for arr in final_outlines if len(arr) > 0] # cleaning contour array (removing any empty arrays resulted from user interaction)
    # flattened = [item for sublist in final_outlines for item in sublist] # currently stores sets of coordinates per highlighted path - have to make it a 1d array of contour coordinates
    evaluation(contour_img, eval_img_path, spreadsheet_path)