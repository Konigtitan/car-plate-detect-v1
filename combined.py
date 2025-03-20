import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import string
import time
import asyncio
from PIL import Image
from ultralytics import YOLO

# Ensure the event loop is properly initialized for async operations
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import EasyOCR for license plate recognition
import easyocr

# Set page config for better layout
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide"
)

# Initialize the OCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion in license plates
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop, reader):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

# Load YOLO models with error handling
@st.cache_resource
def load_license_plate_model(model_path='./models/license_plate_detector.pt'):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading license plate model: {e}")
        st.info("Please make sure the license plate model file exists and is compatible with your Ultralytics version.")
        return None

@st.cache_resource
def load_parking_model(model_path='runs/detect/parking_model/weights/best.pt'):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading parking model: {e}")
        st.info("Please make sure the parking model file exists and is compatible with your Ultralytics version.")
        return None

# Process parking detection results
def process_parking_results(results, frame):
    if len(results) == 0:
        return frame, 0, 0, 0

    # Access the first result
    result = results[0]

    # Initialize counters for vacant and occupied spots
    vacant_spots = 0
    occupied_spots = 0

    # Get class mapping from the model
    class_names = result.names

    # Process detections
    boxes = []
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # Get coordinates
            xyxy = box.xyxy[0].tolist()

            # Count based on class ID
            class_name = class_names[cls_id].lower()

            # Check different possible class name variations
            if 'empty' in class_name or 'vacant' in class_name or cls_id == 0:
                vacant_spots += 1
            else:
                occupied_spots += 1

            boxes.append([*xyxy, conf, cls_id])

    # Calculate total spots
    total_spots = vacant_spots + occupied_spots

    # Draw bounding boxes and labels on the frame
    for det in boxes:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Determine class name and color
        class_name = class_names[cls_id]
        is_vacant = 'empty' in class_name.lower() or 'vacant' in class_name.lower() or cls_id == 0
        color = (0, 255, 0) if is_vacant else (0, 0, 255)  # Green for vacant, Red for occupied

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{'Vacant' if is_vacant else 'Occupied'}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the frame to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = Image.fromarray(frame_rgb)

    return processed_frame, vacant_spots, occupied_spots, total_spots

# Process the image for license plate detection
def process_license_plate_image(model, image, reader):
    if model is None:
        return image
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Detect license plates
    license_plates = model(image_np)[0]
    
    # Process each detected license plate
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop license plate
        license_plate_crop = image_np[int(y1):int(y2), int(x1):int(x2), :]
        
        # Process license plate
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop, reader)
        
        if license_plate_text is not None:
            # Draw bounding box and text
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_np, license_plate_text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Process a video file or webcam stream with combined detection
def process_combined_video(license_plate_model, parking_model, video_path, confidence_threshold, reader, 
                           stop_button, pause_button, detection_mode):
    if (detection_mode in ["license_plate", "combined"] and license_plate_model is None) or \
       (detection_mode in ["parking", "combined"] and parking_model is None):
        st.error("Required model not loaded.")
        return

    # Open the video file or camera
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        st.error("Failed to open the video or camera.")
        return

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    st.sidebar.text(f"Resolution: {frame_width}x{frame_height}")
    st.sidebar.text(f"FPS: {fps:.1f}")

    # Placeholder for the video frame
    video_placeholder = st.empty()

    # Initialize performance metrics
    frame_count = 0
    last_update_time = time.time()
    last_fps = 0
    processing_time = 0

    # Create placeholders for metrics
    parking_metrics = st.sidebar.empty()
    performance_metrics = st.sidebar.empty()
    plates_detected = []
    plates_container = st.sidebar.container()

    while True:
        if stop_button:
            break

        if not pause_button:
            ret, frame = video_capture.read()
            if not ret:
                st.warning("End of video or failed to read frame.")
                break

            # Increase frame counter
            frame_count += 1

            # Start timing for this frame
            start_process = time.time()

            # Resize the frame for processing if needed
            resized_frame = cv2.resize(frame, (640, 640))
            
            # Process based on mode
            if detection_mode in ["parking", "combined"]:
                # Perform parking spot detection
                results = parking_model(resized_frame, conf=confidence_threshold, verbose=False)
                processed_frame, vacant_count, occupied_count, total_count = process_parking_results(results, resized_frame.copy())
                
                # Update parking metrics every 10 frames
                if frame_count % 10 == 0:
                    parking_metrics.text(f"Vacant: {vacant_count}/{total_count} spaces")
                    parking_metrics.text(f"Occupied: {occupied_count}/{total_count} spaces")
                
                # Convert processed frame back to numpy for license plate detection if needed
                if detection_mode == "combined":
                    frame = np.array(processed_frame)
            
            if detection_mode in ["license_plate", "combined"]:
                # Perform license plate detection on original frame
                license_plates = license_plate_model(frame)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    # Process license plate
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop, reader)
                    
                    if license_plate_text is not None:
                        # Draw bounding box and text
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Add to detected plates if not already in list
                        if license_plate_text not in plates_detected:
                            plates_detected.append(license_plate_text)
                            # Update detected plates list
                            plates_container.text("Detected License Plates:")
                            for i, plate in enumerate(plates_detected[-10:], 1):  # Show most recent 10
                                plates_container.text(f"{i}. {plate}")

            # Calculate processing time
            processing_time = (time.time() - start_process) * 1000  # in ms

            # Calculate and update FPS every 10 frames
            if frame_count % 10 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_update_time
                last_fps = 10 / elapsed_time if elapsed_time > 0 else 0
                last_update_time = current_time

                # Update performance metrics
                performance_metrics.text(f"FPS: {last_fps:.1f}")
                performance_metrics.text(f"Processing Time: {processing_time:.1f}ms")

            # Convert to RGB for display
            if detection_mode == "parking" and 'processed_frame' in locals():
                # Use the already processed frame from parking detection
                display_frame = processed_frame
            else:
                # Convert the frame with license plates to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = Image.fromarray(frame_rgb)

            # Display the processed frame
            video_placeholder.image(display_frame, use_container_width=True)

    # Release the video capture object
    video_capture.release()

# Main Streamlit app
def main():
    st.title("Vehicle Detection System")
    
    # Load models
    reader = load_ocr_reader()
    
    # Create a tab layout
    tab1, tab2 = st.tabs(["Main Interface", "About"])
    
    with tab1:
        # Set up the layout with columns
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.sidebar.title("Settings")
            
            # Model selection
            st.sidebar.subheader("Model Settings")
            detection_mode = st.sidebar.radio(
                "Detection Mode", 
                ["license_plate", "parking", "combined"],
                format_func=lambda x: {"license_plate": "License Plate Detection", 
                                      "parking": "Parking Spot Detection", 
                                      "combined": "Combined Detection"}[x]
            )
            
            confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.01)
            
            # Environment info
            st.sidebar.subheader("Environment Info")
            try:
                import ultralytics
                st.sidebar.info(f"Ultralytics version: {ultralytics.__version__}")
            except:
                st.sidebar.warning("Could not detect Ultralytics version")
            
            # Input source selection
            st.sidebar.subheader("Input Source")
            input_source = st.sidebar.radio("Select input source", ["Upload File", "Use Webcam"])
            
        with col1:
            # Handle file upload or webcam selection
            if input_source == "Upload File":
                uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
                
                if uploaded_file is not None:
                    # Save the uploaded file to a temporary location
                    file_path = os.path.join(os.getcwd(), uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process based on file type
                    if uploaded_file.type.startswith("image"):
                        # Load image
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        
                        # Load appropriate models based on detection mode
                        license_plate_model = None
                        parking_model = None
                        
                        if detection_mode in ["license_plate", "combined"]:
                            license_plate_model = load_license_plate_model()
                        
                        if detection_mode in ["parking", "combined"]:
                            parking_model = load_parking_model()
                        
                        # Process button
                        if st.button("Process Image"):
                            with st.spinner("Processing..."):
                                image_np = np.array(image)
                                
                                # Apply appropriate processing based on mode
                                if detection_mode == "license_plate":
                                    processed_image = process_license_plate_image(license_plate_model, image_np, reader)
                                    st.image(processed_image, caption="Processed Image", use_container_width=True)
                                    
                                elif detection_mode == "parking":
                                    results = parking_model(image_np, conf=confidence_threshold, verbose=False)
                                    processed_frame, vacant_count, occupied_count, total_count = process_parking_results(results, image_np.copy())
                                    st.image(processed_frame, caption="Processed Image", use_container_width=True)
                                    st.sidebar.text(f"Vacant: {vacant_count}/{total_count} spaces")
                                    st.sidebar.text(f"Occupied: {occupied_count}/{total_count} spaces")
                                    
                                elif detection_mode == "combined":
                                    # First do parking detection
                                    results = parking_model(image_np, conf=confidence_threshold, verbose=False)
                                    processed_frame, vacant_count, occupied_count, total_count = process_parking_results(results, image_np.copy())
                                    
                                    # Then do license plate detection
                                    final_image = process_license_plate_image(license_plate_model, np.array(processed_frame), reader)
                                    
                                    st.image(final_image, caption="Processed Image", use_container_width=True)
                                    st.sidebar.text(f"Vacant: {vacant_count}/{total_count} spaces")
                                    st.sidebar.text(f"Occupied: {occupied_count}/{total_count} spaces")
                        
                    elif uploaded_file.type.startswith("video"):
                        # Load appropriate models based on detection mode
                        license_plate_model = None
                        parking_model = None
                        
                        if detection_mode in ["license_plate", "combined"]:
                            license_plate_model = load_license_plate_model()
                        
                        if detection_mode in ["parking", "combined"]:
                            parking_model = load_parking_model()
                        
                        # Video controls
                        st.subheader("Video Controls")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            start_button = st.button("Start Processing")
                        with col2:
                            stop_button = st.button("Stop")
                        with col3:
                            pause_button = st.button("Pause/Resume")
                        
                        if start_button:
                            process_combined_video(license_plate_model, parking_model, file_path, 
                                                confidence_threshold, reader, stop_button, pause_button, detection_mode)
                        
                    # Clean up the temporary file
                    os.remove(file_path)
            
            else:  # Webcam option
                # Load appropriate models based on detection mode
                license_plate_model = None
                parking_model = None
                
                if detection_mode in ["license_plate", "combined"]:
                    license_plate_model = load_license_plate_model()
                
                if detection_mode in ["parking", "combined"]:
                    parking_model = load_parking_model()
                
                # Webcam controls
                st.subheader("Webcam Controls")
                col1, col2, col3 = st.columns(3)
                with col1:
                    start_cam_button = st.button("Start Camera")
                with col2:
                    stop_cam_button = st.button("Stop Camera")
                with col3:
                    pause_cam_button = st.button("Pause/Resume Camera")
                
                if start_cam_button:
                    process_combined_video(license_plate_model, parking_model, 0, 
                                        confidence_threshold, reader, stop_cam_button, pause_cam_button, detection_mode)
    
    with tab2:
        st.header("About This Application")
        st.write("""
        This integrated system combines two powerful detection capabilities:
        
        1. **License Plate Detection**: Identifies and reads license plates from vehicles using YOLO object detection and OCR.
        
        2. **Parking Spot Detection**: Analyzes parking lots to identify vacant and occupied parking spaces.
        
        3. **Combined Mode**: Utilizes both detection systems simultaneously for comprehensive vehicle monitoring.
        
        ### Usage Instructions:
        
        1. Select the desired detection mode from the sidebar
        2. Choose your input source (uploaded file or webcam)
        3. Adjust confidence threshold as needed
        4. Process your content and view the results
        
        ### Requirements:
        
        - Ultralytics YOLOv8
        - EasyOCR
        - Pre-trained models for license plate and parking spot detection
        """)

if __name__ == "__main__":
    main()