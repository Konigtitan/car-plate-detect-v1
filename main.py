import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import string  # Required for string.ascii_uppercase

# Load the ultralytics patch
from ultralytics_patch import get_yolo_model
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
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

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

# Try to load the model with error handling
@st.cache_resource
def load_model():
    try:
        return get_yolo_model('./models/license_plate_detector.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model file exists and is compatible with your Ultralytics version.")
        return None

def main():
    st.title("Car License Plate Detection")
    st.sidebar.title("Settings")
    st.sidebar.markdown("Choose the input source and adjust parameters.")

    # Check if model version is compatible
    st.sidebar.markdown("### Environment Info")
    try:
        import ultralytics
        st.sidebar.info(f"Ultralytics version: {ultralytics.__version__}")
    except:
        st.sidebar.warning("Could not detect Ultralytics version")

    input_source = st.sidebar.radio("Select input source", ["Upload MP4", "Use Webcam"])

    if input_source == "Upload MP4":
        uploaded_file = st.sidebar.file_uploader("Upload an MP4 file", type=["mp4"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            st.warning("Please upload an MP4 file.")
            return
    else:
        video_path = 0  # Webcam

    # Load the license plate detector model
    license_plate_detector = load_model()
    
    # Check if model loaded successfully
    if license_plate_detector is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video source.")
        return

    stframe = st.empty()
    stop_button = st.sidebar.button("Stop")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            # Process license plate
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
            
            if license_plate_text is not None:
                # Draw bounding box and text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    if input_source == "Upload MP4":
        os.unlink(video_path)

if __name__ == "__main__":
    main()