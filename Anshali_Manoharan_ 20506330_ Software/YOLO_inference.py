from ultralytics import YOLO
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import time

def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def identify_object_from_contour(frame, x_coords, y_coords, model, iou_threshold=0.5):
    # Form bounding box from contour
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    contour_box = [x_min, y_min, x_max, y_max]

    result = model(frame)[0]

    for box in result.boxes:# for every bounding box identified by YOLO
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_box = [x1, y1, x2, y2]
        iou = compute_iou(contour_box, yolo_box)

        if iou > iou_threshold:
            class_id = int(box.cls)
            return class_id

    return None

def get_YOLO_model(path = "yolo11n.pt"):
    return YOLO(path)

def get_bbox_center(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return [center_x, center_y]

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def evaluate_predicted_contours(model, eval_data, target_id):
    iou_scores = []
    center_errors = []
    total_time = 0
    frame_count = 0

    for frame, contour in eval_data:
        start_time = time.time()
        x_coords = contour[:, 0]
        y_coords = contour[:, 1]
        contour_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        result = model(frame)[0]
        boxes = result.boxes

        best_iou = -1
        best_box = None

        for box in boxes: # for every detected box in the current frame, identify the closest one to the predicted contour with same ID
            class_id = int(box.cls)
            if class_id != target_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_box = [x1, y1, x2, y2]
            iou = compute_iou(contour_box, yolo_box)

            if iou > best_iou:
                best_iou = iou
                best_box = yolo_box

        iou_scores.append(best_iou)

        # CONDENSATION contour box (green)
        cv2.rectangle(frame, (contour_box[0], contour_box[1]), (contour_box[2], contour_box[3]), (0, 255, 0), 2)
        # best matching YOLO box (red)
        if best_box:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 0, 255), 2)

        # display IoU
        label = f"IoU: {best_iou:.2f}" if best_iou >= 0 else "No Match Found"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display frame
        cv2.imshow("Evaluation", frame)

        center1 = get_bbox_center(best_box[0], best_box[1], best_box[2], best_box[3])
        center2 = get_bbox_center(contour_box[0], contour_box[1], contour_box[2], contour_box[3])

        center_errors.append(euclidean_distance(center1, center2))


        frame_time = time.time() - start_time
        total_time += frame_time
        frame_count += 1

        key = cv2.waitKey(30)  # press Esc to stop early
        if key == 27:
            break

    cv2.destroyAllWindows()

    if total_time > 0:
        fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds — FPS: {fps:.2f}")

    fig = plt.figure(figsize=(10, 5))
    fig.canvas.manager.set_window_title("Evaluation Results")
    plt.plot(range(len(iou_scores)), iou_scores, marker='.', linestyle='-', color='magenta')
    plt.title(f"IoU over Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("IoU Score")
    plt.grid(True)
    plt.ylim(0, 1)
    mean_center_error = np.mean(center_errors)
    plt.figtext(0.01, 0.01, f"Mean Center Error: {mean_center_error:.2f}", ha="left", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for bottom text
    plt.show()

    return


