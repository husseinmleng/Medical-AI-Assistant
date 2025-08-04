# yolo_model.py
from ultralytics import YOLO
import os
import uuid
from PIL import Image

# Load the trained YOLOv8 model
# Create an absolute path to the model file to avoid any pathing issues.
model_path = r"/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/src/weights/best.pt"
model = YOLO(model_path)

# Ensure the directory for annotated images exists
output_dir = "annotated_images"
os.makedirs(output_dir, exist_ok=True)

def detect_cancer_in_image(image_path: str):
    """
    Analyzes a medical image using the YOLO model to detect signs of cancer.

    Args:
        image_path (str): The path to the user's uploaded image.

    Returns:
        A tuple containing:
        - result_text (str): "Positive" or "Negative".
        - confidence (float): The confidence score of the detection (0.0 to 1.0).
        - annotated_image_path (str or None): The path to the saved annotated image, or None.
    """
    try:
        # Run inference on the image
        results = model(image_path, conf=0.5)  # Use a confidence threshold of 0.5

        result_text = "Negative"
        confidence = 0.0
        has_detection = False
        annotated_image_path = None

        for r in results:
            if len(r.boxes) > 0:
                has_detection = True
                # Find the detection with the highest confidence for the 'cancer' class
                max_conf = 0.0
                cancer_detected = False
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name.lower() == 'cancer':
                        cancer_detected = True
                        if box.conf[0] > max_conf:
                            max_conf = float(box.conf[0])
                
                if cancer_detected:
                    result_text = "Positive"
                    confidence = max_conf

        if has_detection:
            # Generate a unique filename for the annotated image
            unique_filename = f"{uuid.uuid4()}.png"
            save_path = os.path.join(output_dir, unique_filename)

            # Save the annotated image
            annotated_image = Image.fromarray(results[0].plot()[..., ::-1])
            annotated_image.save(save_path)
            annotated_image_path = save_path
            
            print(f"YOLO Analysis Result: {result_text}, Confidence: {confidence:.2f}, Annotated image saved to: {annotated_image_path}")
        else:
            print("YOLO Analysis Result: Negative (No detections)")

        return result_text, confidence, annotated_image_path

    except Exception as e:
        print(f"Error during YOLO model inference: {e}")
        return "Error", 0.0, None
