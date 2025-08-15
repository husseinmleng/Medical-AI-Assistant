from ultralytics import YOLO
import torch
import os
import uuid
from PIL import Image

# --- FIX: Use relative paths and create output directory robustly ---
_script_dir = os.path.dirname(__file__)
_project_root = os.path.dirname(_script_dir) # Assumes src is in project root

# Model Path
model_path = os.path.join(_script_dir, "weights", "best.pt")
model = YOLO(model_path)

# Select device with safe CPU fallback
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory for annotated images
# Place it in the project root to be easily accessible by Streamlit
output_dir = os.path.join(_project_root, "annotated_images")
os.makedirs(output_dir, exist_ok=True)

def detect_cancer_in_image(image_path: str) :
    """
    Analyzes a medical image using the YOLO model to detect signs of cancer.

    Args:
        image_path (str): The path to the user's uploaded image.

    Returns:
        A tuple containing:
        - result_text (str): "Positive" or "Negative".
        - confidence (float): The confidence score of the detection (0.0 to 1.0).
        - annotated_image_path (str or None): The absolute path to the saved annotated image, or None.
    """
    try:
        # Run inference on the provided image_path with robust device selection
        try:
            results = model.predict(image_path, conf=0.5, device=_device)
        except Exception as e:
            # Fallback to CPU if CUDA is busy/unavailable at runtime
            print(f"YOLO inference failed on device '{_device}' with error: {e}. Falling back to CPU.")
            results = model.predict(image_path, conf=0.5, device="cpu")

        result_text = "Negative"
        confidence = 0.0
        has_detection = False
        annotated_image_path = None

        # Check all results for detections
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
                        print('-' * 20)
                        print(f"Detected cancer with confidence: {box.conf[0]}")
                        if box.conf[0] > max_conf:
                            max_conf = float(box.conf[0])
                            print('-' * 20)
                            print(f"New max confidence for cancer detection: {max_conf:.4f}")
                    else:
                        print(f"Detected non-cancer class '{class_name}' with confidence: {box.conf[0]}")
                        max_conf = max(max_conf, float(box.conf[0]))

                if cancer_detected:
                    result_text = "Positive"
                    confidence = max_conf
                else:
                    result_text = "Negative"
                    confidence = max_conf if max_conf > 0 else 0.0
        print('-' * 20)
        print(f"Final detection result: {result_text} with confidence: {confidence:.4f}")
        # Save the annotated image ONLY if a detection was made
        if has_detection:
            # Generate a unique filename for the annotated image
            unique_filename = f"{uuid.uuid4()}.png"
            # The save_path must be absolute for Streamlit to find it reliably
            save_path = os.path.join(output_dir, unique_filename)

            # Save the annotated image from the first result object
            annotated_image = Image.fromarray(results[0].plot()[..., ::-1])
            annotated_image.save(save_path)
            annotated_image_path = save_path # This is now an absolute path
            
            print(f"YOLO Analysis Result: {result_text}, Confidence: {confidence:.4f}, Annotated image saved to: {annotated_image_path}")
        else:
            # If no detections, confidence remains 0 for 'Positive'
            # We can assign a high confidence for 'Negative' if needed, but it's simpler this way.
            confidence = 0.0 # Confidence in the "Positive" result is 0
            print("YOLO Analysis Result: Negative (No detections)")

        return result_text, confidence, annotated_image_path

    except Exception as e:
        print(f"Error during YOLO model inference: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, None
