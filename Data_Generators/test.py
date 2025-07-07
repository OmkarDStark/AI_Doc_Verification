import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def plot_ground_truth(image_path, label_path):
    """
    Plots the ground-truth bounding boxes from the label file on the given image.
    """
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read label file and draw boxes
    with open(label_path, "r") as file:
        for line in file:
            data = line.strip().split()
            cls, cx, cy, w, h = map(float, data)

            # Convert normalized coordinates (cx, cy, w, h) to pixel coordinates
            height, width, _ = img.shape
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            # Draw the bounding box (green for ground-truth)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "tumor (GT)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Ground Truth")
    plt.show()

    return img


def test_model(image_path, model_path="/content/liver_tumor_segmentation_model.pt"):
    """
    Uses the trained YOLOv8 model to predict bounding boxes on the given image.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(source=image_path, save=True)

    # Check if image0.jpg is created
    result_img_path = "/content/runs/detect/predict4/image-0033_jpg.rf.72c970c29fbe513a8c51b5b7012d0dec.jpg"
    if not os.path.exists(result_img_path):
        print(f"Error: {result_img_path} does not exist.")
        return

    # Visualize predictions
    results_img = cv2.imread(result_img_path)
    if results_img is None:
        print(f"Error: Failed to read {result_img_path}.")
        return

    results_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(results_img)
    plt.axis("off")
    plt.title("Predictions")
    plt.show()



# Paths to image and label file
image_path = "/content/Liver-Tumor-Detection-2/valid/images/image-0033_jpg.rf.72c970c29fbe513a8c51b5b7012d0dec.jpg"  # Replace with the image path
label_path = "/content/Liver-Tumor-Detection-2/valid/labels/image-0033_jpg.rf.72c970c29fbe513a8c51b5b7012d0dec.txt"  # Replace with the corresponding label file path

# 1. Plot Ground Truth
gt_img = plot_ground_truth(image_path, label_path)

# 2. Test the Model
test_model(image_path)
