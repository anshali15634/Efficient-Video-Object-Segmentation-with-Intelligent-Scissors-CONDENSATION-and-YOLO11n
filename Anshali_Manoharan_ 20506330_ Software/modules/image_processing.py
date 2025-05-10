# modules/image_processing.py
import cv2
import numpy as np

def setup_image(file_path):
    """
    Reads the image file or extracts the first frame from a video file,
    converts it to grayscale, and returns the image and its grayscale version.
    """
    # Read the file based on extension
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read image file.")
        blue_channel, green_channel, red_channel = cv2.split(img)
        gray = np.maximum(np.maximum(red_channel, green_channel), blue_channel)

    elif file_path.lower().endswith('.mp4'):
        video_capture = cv2.VideoCapture(file_path)
        success, frame = video_capture.read()  # Read the first frame
        if not success:
            raise ValueError("Failed to read video file.")
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        video_capture.release()  # Release the video capture object

    else:
        raise ValueError("Unsupported file type. Please select an image or an .mp4 video.")
    blurred_gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    return img, blurred_gray

def setup_gradient_direction(gray):

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # Gradient in x-direction
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # Gradient in y-direction

    g_x = gray * Ix
    g_y = gray * Iy

    #gradient_magnitude =  np.sqrt(g_x**2 + g_y**2)
    gradient_direction = np.arctan2(g_x, g_y)

    # normalise both to range 0 to 1
    #gradient_magnitude /= gradient_magnitude.max() 
    #gradient_direction = (gradient_direction + np.pi/2) / np.pi
    gradient_direction = (gradient_direction + np.pi) / (2 * np.pi) # normalises to the range 0-1
    return gradient_direction.astype(np.float32)

def setup_gradient_magnitude(gray):

    smoothed_3x3 = cv2.GaussianBlur(gray, (3, 3), 0)
    smoothed_5x5 = cv2.GaussianBlur(gray, (5, 5), 0)
    smoothed_7x7 = cv2.GaussianBlur(gray, (7, 7), 0)

    smoothed_img_collection = [smoothed_3x3,smoothed_5x5,smoothed_7x7]
    grad_magnitude_img_collection = [] # for different kernels, store the partial derivatives here.

    # forming gradient magnitude images from different kernels for later use.
    for smoothed_img in smoothed_img_collection:
        sobel_x = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0) # horizontal gradient (x-direction)
        sobel_y = cv2.Sobel(smoothed_img, cv2.CV_64F, 0, 1) # vertical gradient (y-direction)
        sobel_x_squared = np.square(sobel_x)
        sobel_y_squared = np.square(sobel_y)
        sobel_sum = sobel_x_squared + sobel_y_squared
        G = np.sqrt(sobel_sum) # gradient magnitude G
        grad_magnitude_img_collection.append(G)

    # to identify the largest gradient magnitude from the three kernels and store in one empty image
    max_gradient_magnitude_img = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint64)

    # form multi-scale gradient magnitude image
    for row in range(max_gradient_magnitude_img.shape[0]):
        for column in range(max_gradient_magnitude_img.shape[1]):
            max_magnitude = 0
            for kernel_img in grad_magnitude_img_collection:
                if kernel_img[row][column] > max_magnitude:
                    max_magnitude = kernel_img[row][column]
            max_gradient_magnitude_img[row][column] = max_magnitude

    # normalise final image
    min_gradient_value = np.min(max_gradient_magnitude_img) # min(G)
    Gdash = max_gradient_magnitude_img - min_gradient_value
    G_dash_max = np.max(Gdash)
    final_gradient_magnitude = 1 - (Gdash/(G_dash_max)) # values range from 0.0 to 1.0
    return final_gradient_magnitude.astype(np.float32)

def setup_canny(gray):
    canny = cv2.Canny(gray.astype(np.uint8), 100, 200) # default values are 100 and 200 for maximal supression, canny values from 0 - 255
    canny = cv2.bitwise_not(canny) # invert values of canny image so that edges have low cost
    canny = (canny // 255)  # Normalize and convert to float32 - normalise 0-255 pixel value range to 0.0 - 1.0 range
    return canny.astype(np.float32)