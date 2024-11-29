import cv2
import pytesseract
import re
import numpy as np

# Specify the Tesseract path if not already in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_cascade(file_path):
    """Load Haar cascade for license plate detection."""
    cascade = cv2.CascadeClassifier(file_path)
    if cascade.empty():
        print(f"Error loading cascade from {file_path}")
        return None
    return cascade

def preprocess_image(plate_roi):
    """Preprocess the plate region for better OCR recognition."""
    # Check if the image has 3 channels (color image)
    if len(plate_roi.shape) == 3 and plate_roi.shape[2] == 3:
        # Convert to grayscale if it's a color image
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    elif len(plate_roi.shape) == 2:
        # It's already grayscale
        gray = plate_roi
    else:
        print("Invalid plate region: too few channels.")
        return None
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def validate_plate(text):
    """Validate the plate text to match Philippine old or new format."""
    # Old format (3 letters - 3 numbers): ABC-123
    pattern_old = r'^[A-Z]{3}-\d{3}$'  # Matching 'ABC-123'
    # New format (3 letters - 4 numbers): ABC-1234
    pattern_new = r'^[A-Z]{3}-\d{4}$'  # Matching 'ABC-1234'
    
    # Return True if either format matches
    return re.match(pattern_old, text) or re.match(pattern_new, text)

def preprocess_image(plate_roi):
    """Preprocess the plate region for better OCR recognition."""
    # Check if the plate_roi is valid
    if plate_roi is None or plate_roi.size == 0:
        print("Error: Plate ROI is empty or None")
        return None
    
    # Convert to grayscale if it's a color image
    if len(plate_roi.shape) == 3 and plate_roi.shape[2] == 3:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    elif len(plate_roi.shape) == 2:
        gray = plate_roi
    else:
        print("Invalid plate region: too few channels.")
        return None

    # Increase contrast using histogram equalization
    contrast_enhanced = cv2.equalizeHist(gray)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)

    # Apply binary thresholding for clearer edges (using Otsu's method)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Apply morphological transformations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

def detect_philippine_plate(frame, reference_plate_image=None):
    """Detect and recognize Philippine license plates using DoG and SIFT."""
    
    # Convert image to grayscale for SIFT feature extraction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and histogram equalization for better detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)

    # Load Haar Cascade for detecting the license plate
    plate_cascade = load_cascade('haarcascade_russian_plate_number.xml')  # Adjust for more accurate cascade if needed
    if plate_cascade is None:
        print("Failed to load plate cascade. Exiting.")
        return frame

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(equalized, scaleFactor=1.05, minNeighbors=5, minSize=(100, 50))

    if len(plates) == 0:
        print("No plates detected.")
        return frame

    # Loop through detected plates
    for (x, y, w, h) in plates:
        # Draw a rectangle around the detected plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the detected plate region (ROI)
        plate_roi = equalized[y:y + h, x:x + w]
        
        # Preprocess the plate region
        preprocessed_plate = preprocess_image(plate_roi)
        
        # If preprocessing failed, skip this plate
        if preprocessed_plate is None:
            continue
        
        # Perform text recognition using pytesseract for OCR
        plate_text = pytesseract.image_to_string(preprocessed_plate, config='--psm 8')
        plate_text = ''.join(filter(str.isalnum, plate_text)).upper()

        # Validate the detected plate text
        if validate_plate(plate_text):
            print(f"Valid Plate Detected: {plate_text}")
            # Draw the validated plate text on the image
            cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print(f"Invalid or Unreadable Plate: {plate_text}")
    
    return frame


# Example Usage (outside the module)
if __name__ == "__main__":
    # Load a reference plate image (this could be an image of a known plate)
    reference_plate = cv2.imread('reference.png')  # Change this to the path of your reference plate image

    # Option 1: Use webcam feed for plate detection
    cap = cv2.VideoCapture(0)  # Using webcam (you can replace with video file path)

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            print("Failed to capture image")
            break

        # Detect plate using SIFT and DoG
        result_frame = detect_philippine_plate(frame, reference_plate_image=reference_plate)

        # Display the result frame
        cv2.imshow("Detected License Plate", result_frame)

        # Exit on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
