# necessary functions for observation density
import cv2
import numpy as np
import matplotlib.pyplot as plt

# helper functions
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    
    return np.array(frames, dtype=np.uint8)

def generate_initial_state_vectors(numberOfSamples, translation_range=(-70, 70)):
    # generates initial sample set / set of particles/samples
    
    state_vectors = np.zeros((numberOfSamples, 2))  # Initialize an array for the state vectors
    
    for i in range(numberOfSamples):
        state_vectors[i, 0] = np.random.uniform(*translation_range)  # x-translation
        state_vectors[i, 1] = np.random.uniform(*translation_range)  # y-translation
    return state_vectors

def generate_random_state_vector(mean_transformation, translation_range):
    # this set of random large state vectors around the current mean to help in recovery of tracking!
    return [np.random.uniform(-translation_range, translation_range)+mean_transformation[0], np.random.uniform(-translation_range, translation_range)+mean_transformation[1]]

def process_video_frame(frame):
    # convert to given frame convert to canny
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame to grayscale
    smoothed_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0) #gaussian smoothing, default kernel size (5x5)
    edges = cv2.Canny(smoothed_frame, 100, 200)  # canny edge detection, default thresholds (100, 200)
    
    return edges # return edge-detected binary image

def get_transformed_contour(object_contour, state_vector):
    # convert this into one large matrix after it works properly.
    transformed_contour = []
    tx, ty = state_vector[0], state_vector[1]
    for x, y in object_contour:
        x_transformed = x + tx
        y_transformed = y + ty
        transformed_contour.append((int(x_transformed), int(y_transformed)))
    return np.array(transformed_contour)

def coordinate_subset_indices(transformed_object_contour, num_points=12):

    # returns a subset of coordinate indices which are evenly spaced in the original contour
    
    total_points = len(transformed_object_contour)
    if num_points > total_points:
        raise ValueError("Number of points to select exceeds available contour points.")
    step_size = max(1, total_points // num_points)
    selected_indices = [i for i in range(0, total_points, step_size)][:num_points]
    return selected_indices

def plot_transformed_contours(transformed_contours, img_shape):
    plt.figure()
    # different color for each contour
    cmap = plt.cm.get_cmap("hsv", len(transformed_contours))
    for i, contour in enumerate(transformed_contours):
        contour = np.array(contour) # easy indexing
        plt.plot(contour[:, 0], contour[:, 1], color=cmap(i), linewidth=0.5)

    plt.xlim(0, img_shape[1])
    plt.ylim(0, img_shape[0])
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.title("Transformed Contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.show()

def is_within_bounds(coords, img_shape):
    # for checking the contours if they are within the boundaries of image dimensions
    h, w = img_shape[:2]
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    x_valid = (x_coords >= 0) & (x_coords < w)
    y_valid = (y_coords >= 0) & (y_coords < h)
    return np.all(x_valid & y_valid)

def average_difference1(transformed_contour, edges, selected_indices, sigma=3):

    # smaller sigma values penalise larger distances from actual object contour
    # compute the distance transform (directly using edges)

    dist_transform = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)

    distances = []

    for x,y in transformed_contour: # for all the contour coordinates check the nearest edge!
        if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
            distance = dist_transform[y, x]  # get distance from precomputed map
        else:
            continue
        distances.append(distance)


    # compute Gaussian-weighted sum and total distance sum
    gaussian_sum = np.sum(np.exp(-np.square(distances) / (2 * sigma**2)))

    return gaussian_sum

def systematicResampling(weightArray):

    # returns N array of indices of the new set of states, derived from the previous set of states
    # weights represent the probability of its respective state being chosen

    N = len(weightArray) # systematic resampling is like using a roulette wheel, N is total no. of samples
    cValues = [] # cumulative sum of weights
    cValues.append(weightArray[0])
    for i in range(N-1):
        cValues.append(cValues[i]+weightArray[i+1])
    startingPoint = np.random.uniform(low=0.0, high = 1/N) # since total probability = 1
    resampledIndex = [] # will store the newly drawn N number of samples

    for j in range(N):
        currentPoint = startingPoint + (1/N)*j
        s=0
        while (currentPoint>cValues[s]): # check which boundary is overlapped to identify the sample drawn
            s = s+1
        resampledIndex.append(s)
    return resampledIndex # stores the index of the particles chosen

# process video
def CONDENSATION_main(video_path, object_contour, initial_frame_obj_mask):

    final_contours_evaluation = []
    frames = extract_frames(video_path)
    # defining parameters
    numberOfSamples = 75
    numberOfFrames = len(frames) # this should be the number of frames in video - dictates number of iterations of particle filter
    A = 0.4 # defines weight of state during stochastic diffusion
    width = [1.5, 1.5]
    width = np.array(width)
    total_replaced = 10 # the number of samples to be replaced after resampling in case we need recovery during tracking
    gaussian_diffusion = 5
    prev_check = 0 # stores accuracy check from previous frame

    # initialisation
    new_states = generate_initial_state_vectors(numberOfSamples) # list used to store the states and track convergence 
    new_weights = np.ones(numberOfSamples) / numberOfSamples # store weights and track convergence (initial weight set is equal probabilities for all samples)
    mean_state_vector = np.array([0,0]) #np.sum(new_states * new_weights[:, np.newaxis], axis = 0)
    final_transformed_contour = None

    plt.ion()

    # start CONDENSATION algorithm
    for i in range(1, int(numberOfFrames)): # as the first frame was used to get the contour from user.

        # propagation
        for k in range(len(new_states)):
            gaussian_term = np.random.normal(0, gaussian_diffusion, size=new_states[k].shape)
            new_states[k] = mean_state_vector + A * (new_states[k] - mean_state_vector) + width * gaussian_term
        
        # transform before observation probability calculation
        transformed_contours = []
        for g in range(numberOfSamples):
            transformed_contours.append(get_transformed_contour(object_contour, new_states[g]))
        
        # observation probability calculated
        edges = process_video_frame(frames[i]) # get edge detection of the current frame
        new_weights = np.zeros(numberOfSamples)
        
        for j in range(numberOfSamples):
            transformed_contour = transformed_contours[j]
            selected_indices = coordinate_subset_indices(transformed_contour)
            new_weights[j] = average_difference1(transformed_contour, edges, selected_indices)
            # new_weights[j] = average_difference2(transformed_contour, initial_frame_obj_mask, frames[i], frames[0])

        new_weights /= new_weights.sum() # normalise the weights to add up to 1

        best_contour = new_states[np.argmax(new_weights)]

        # resampling
        resampledStateIndices=systematicResampling(new_weights) # outputs the indices of states to replace old ones
        # updating states and weights
        new_states_updated = np.zeros(new_states.shape)
        new_weights_updated = np.zeros(new_weights.shape)
        count = 0

        for f in resampledStateIndices:
            new_states_updated[count] = new_states[f]
            new_weights_updated[count] = new_weights[f]
            count+=1
        
        new_states = np.array(new_states_updated) # updated states and weights

        # here replace lowest weighted weights with random samples for recovery in tracking later :D
        lowest_indices = np.argpartition(new_weights_updated, total_replaced)[:total_replaced]
        for index in lowest_indices:
            new_states[index] = generate_random_state_vector(best_contour, translation_range=70)
        new_weights = np.ones(numberOfSamples) / numberOfSamples # after systematic resampling, all weights become equal again.
        
        # calculate expected value of state vector for display
        mean_state_vector= best_contour #np.sum(new_states * new_weights[:, np.newaxis], axis=0)

        final_transformed_contour = get_transformed_contour(object_contour, mean_state_vector)
        final_transformed_contour = np.array(final_transformed_contour)

        # display the frame and the contour over it
        plt.clf()
        plt.gcf().canvas.manager.set_window_title('CONDENSATION') 
        frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        #plt.scatter(x_coords, y_coords, color = 'red', s=0.2)
        plt.scatter(final_transformed_contour[:, 0], final_transformed_contour[:, 1], color='#37FD12', s=0.5) #plt.plot(x_coords, y_coords, color='red', linewidth=2)
        plt.title(f"Frame {i+1}/{numberOfFrames}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.05)

        final_contours_evaluation.append(final_transformed_contour)

    plt.ioff()
    plt.show()

    return final_contours_evaluation, frames[1:]
