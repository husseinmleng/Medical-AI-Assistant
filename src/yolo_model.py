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
        - annotated_image_path (str or None): The path to the saved annotated image, or None.
    """
    try:
        # Run inference on the image
        results = model(image_path, conf=0.5)  # Use a confidence threshold of 0.5

        # --- THIS IS THE FIX ---
        # We need to check the specific class detected, not just if a detection exists.
        result_text = "Negative"  # Default to Negative
        has_detection = False

        for r in results:
            # Check if any boxes were detected
            if len(r.boxes) > 0:
                has_detection = True
                # Iterate through detected boxes to check for the 'cancer' class
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # If the detected class is 'cancer', the result is Positive.
                    if class_name.lower() == 'cancer':
                        result_text = "Positive"
                        break # Exit the loop as soon as cancer is found
            if result_text == "Positive":
                break # Exit the outer loop too

        # --- ANNOTATION LOGIC (REMAINS THE SAME) ---
        annotated_image_path = None
        if has_detection:
            # Generate a unique filename for the annotated image
            unique_filename = f"{uuid.uuid4()}.png"
            save_path = os.path.join(output_dir, unique_filename)

            # Save the annotated image
            annotated_image = Image.fromarray(results[0].plot()[..., ::-1])
            annotated_image.save(save_path)
            annotated_image_path = save_path
            
            print(f"YOLO Analysis Result: {result_text}, Annotated image saved to: {annotated_image_path}")
        else:
            print("YOLO Analysis Result: Negative (No detections)")

        return result_text, annotated_image_path

    except Exception as e:
        print(f"Error during YOLO model inference: {e}")
        return "Error", None
