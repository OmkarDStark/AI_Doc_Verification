from flask import Flask, request, jsonify, render_template_string
import os
import uuid
import re
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import base64
import io
import json
import time

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model map (None for regex-based documents)
DOC_MODEL_PATHS = {
    "Aadhar Card": "/home/stark/Desktop/Document Verification/models/aadhaar.pt",
    "PAN Card": "/home/stark/Desktop/Document Verification/models/pan_best.pt",
    "Handicap Smart Card": "/home/stark/Desktop/Document Verification/models/handicap_smart_card.pt",
    "Birth Certificate": "/home/stark/Desktop/Document Verification/models/Birth_certificatebest.pt",
    "Bonafide Certificate": None,
    "Caste Certificate": "/home/stark/Desktop/Document Verification/models/caste_certificate.pt",
    "Salary Slip": "/home/stark/Desktop/Document Verification/models/salarysliplast.pt",
    "Passport": "/home/stark/Desktop/Document Verification/models/passport.pt",
    "Marksheet": None,
    "Transgender Certificate": "/home/stark/Desktop/Document Verification/models/trans_best.pt"
}

def extract_text_from_bbox(image, bbox, debug=False):
    """
    Extract text from a bounding box region using multiple OCR techniques
    Enhanced version with improved preprocessing and text extraction

    Args:
        image: Original image (RGB format)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        debug: Whether to show intermediate processing steps

    Returns:
        extracted_text: Text found in the bounding box
    """
    x1, y1, x2, y2 = bbox

    # Add more aggressive padding to capture context
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    # Crop the region of interest
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return ""

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Try multiple preprocessing techniques
    preprocessing_methods = []

    # Method 1: Original grayscale
    preprocessing_methods.append(("Original", roi_gray))

    # Method 2: Gaussian blur + threshold
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("OTSU", thresh1))

    # Method 3: Adaptive threshold (multiple variants)
    adaptive_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    preprocessing_methods.append(("Adaptive", adaptive_thresh))
    
    # Method 3b: Adaptive threshold with different parameters
    adaptive_thresh2 = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 15, 5)
    preprocessing_methods.append(("Adaptive2", adaptive_thresh2))

    # Method 4: Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    preprocessing_methods.append(("Morphology", morph))
    
    # Method 4b: Different morphological operations
    kernel2 = np.ones((1, 1), np.uint8)
    morph2 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel2)
    preprocessing_methods.append(("Morphology2", morph2))

    # Method 5: Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("Enhanced", enhanced_thresh))

    # Method 6: Histogram equalization
    hist_eq = cv2.equalizeHist(roi_gray)
    _, hist_thresh = cv2.threshold(hist_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("HistEq", hist_thresh))

    # Method 7: Bilateral filter + threshold
    bilateral = cv2.bilateralFilter(roi_gray, 9, 75, 75)
    _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("Bilateral", bilateral_thresh))

    # Method 8: Upscale image for better OCR (more aggressive)
    height, width = roi_gray.shape
    scale_factor = 2 if height > 30 and width > 30 else 4
    upscaled = cv2.resize(roi_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    _, upscaled_thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("Upscaled", upscaled_thresh))

    # Method 9: Inverted threshold (for white text on dark background)
    _, inverted = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessing_methods.append(("Inverted", inverted))

    # Method 10: Denoising
    denoised = cv2.fastNlMeansDenoising(roi_gray)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(("Denoised", denoised_thresh))

    # Expanded OCR configurations to try
    ocr_configs = [
        '--psm 6',  # Uniform block of text
        '--psm 7',  # Single text line
        '--psm 8',  # Single word
        '--psm 13', # Raw line. Treat the image as a single text line
        '--psm 4',  # Single column of text
        '--psm 3',  # Fully automatic page segmentation
        '--psm 11', # Sparse text
        '--psm 12', # Sparse text with OSD
        '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/',
        '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/',
        '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/',
        '--oem 3 --psm 6',  # Use LSTM engine
        '--oem 1 --psm 6',  # Use original tesseract engine
    ]

    all_texts = []
    best_text = ""
    best_confidence = 0

    # Try each preprocessing method with each OCR config
    for method_name, processed_roi in preprocessing_methods:
        for config in ocr_configs:
            try:
                # Convert to PIL Image
                roi_pil = Image.fromarray(processed_roi)

                # Extract text with confidence
                data = pytesseract.image_to_data(roi_pil, config=config, output_type=pytesseract.Output.DICT)

                # Filter out low confidence detections and collect all text
                confidences = []
                texts = []
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 10:  # Lower threshold for more text
                        confidences.append(int(conf))
                        text_item = data['text'][i].strip()
                        if text_item:
                            texts.append(text_item)

                if confidences and texts:
                    avg_confidence = sum(confidences) / len(confidences)
                    combined_text = ' '.join(texts).strip()

                    # Store all reasonable results
                    if combined_text and avg_confidence > 15:
                        all_texts.append((combined_text, avg_confidence, method_name, config))

                    # Track best result
                    if combined_text and avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_text = combined_text

                        if debug:
                            print(f"Method: {method_name}, Config: {config}")
                            print(f"Text: '{combined_text}', Confidence: {avg_confidence:.2f}")

            except Exception as e:
                continue

    # If no good result, try simple string extraction with multiple methods
    if not best_text or best_confidence < 30:
        fallback_configs = ['--psm 6', '--psm 7', '--psm 8', '--psm 13', '--psm 4']
        for method_name, processed_roi in preprocessing_methods:
            for config in fallback_configs:
                try:
                    roi_pil = Image.fromarray(processed_roi)
                    simple_text = pytesseract.image_to_string(roi_pil, config=config).strip()
                    if simple_text and len(simple_text) > len(best_text):
                        best_text = simple_text
                        all_texts.append((simple_text, 20, method_name, config))
                except:
                    continue

    # Try to combine results from multiple methods for completeness
    if len(all_texts) > 1:
        # Sort by confidence and length
        all_texts.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        # Check if we can combine texts for better results
        top_texts = [text[0] for text in all_texts[:3]]  # Top 3 results
        
        # Find the longest meaningful text
        for text, conf, method, config in all_texts:
            if len(text) > len(best_text) and conf > 15:
                best_text = text
                break

    # Enhanced text cleaning
    if best_text:
        # Remove excessive whitespace and newlines
        best_text = re.sub(r'\s+', ' ', best_text).strip()
        
        # Remove common OCR artifacts
        best_text = re.sub(r'[|\\/*~`]', '', best_text)
        
        # Fix common character substitutions
        char_fixes = {
            '0': 'O', '1': 'l', '5': 'S', '8': 'B', '6': 'G',
            '@': 'a', '#': 'H', '$': 'S', 'I': 'l', 'L': 'i',
            ': ': 'S', '%': 'X'
        }
        
        # Only apply fixes if it seems like it's supposed to be alphabetic
        if best_text and not any(char.isdigit() for char in best_text):
            for wrong, right in char_fixes.items():
                if wrong in best_text and best_text.count(wrong) < 3:  # Don't fix if it appears too often
                    best_text = best_text.replace(wrong, right)
        
        # Remove non-printable characters
        best_text = ''.join(char for char in best_text if char.isprintable())
        
        # Final cleanup
        best_text = best_text.strip()

    return best_text


def load_image(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        images = convert_from_path(file_path)
        return np.array(images[0])
    else:
        return cv2.cvtColor(np.array(Image.open(file_path).convert("RGB")), cv2.COLOR_RGB2BGR)


def run_yolo_ocr(image, model_path):
    model = YOLO(model_path)
    results = model(image)[0]

    fields = {}
    image_drawn = image.copy()
    
    # Convert BGR to RGB for enhanced OCR function
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Special label mapping only for Aadhaar
    aadhaar_label_map = {
        0: "aadhaar_number",
        1: "dob",
        2: "gender",
        3: "name",
        4: "address"
    }

    print(f"🔍 Processing {len(results.boxes)} detected regions...")

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]

        # If Aadhaar model is used, map to human-readable keys
        if "aadhaar" in model_path.lower() and cls_id in aadhaar_label_map:
            class_name = aadhaar_label_map[cls_id]

        print(f"📝 Processing {class_name} (Box {i+1}/{len(results.boxes)})...")

        # Enhanced OCR using integrated function
        text = extract_text_from_bbox(image_rgb, (x1, y1, x2, y2), debug=False)
        
        # Fallback to simple OCR if enhanced method fails
        if not text or text.strip() == "":
            print(f"⚠️ Enhanced OCR failed for {class_name}, trying fallback...")
            try:
                # Fallback: Simple OCR on cropped region
                cropped_rgb = image_rgb[y1:y2, x1:x2]
                if cropped_rgb.size > 0:
                    cropped_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
                    # Try multiple simple approaches
                    fallback_methods = [
                        ('--psm 6', cropped_gray),
                        ('--psm 7', cropped_gray),
                        ('--psm 8', cropped_gray),
                        ('--psm 13', cropped_gray),
                    ]
                    
                    for config, img in fallback_methods:
                        try:
                            fallback_text = pytesseract.image_to_string(
                                Image.fromarray(img), config=config
                            ).strip()
                            if fallback_text and len(fallback_text) > len(text):
                                text = fallback_text
                                break
                        except:
                            continue
            except Exception as e:
                print(f"❌ Fallback OCR also failed for {class_name}: {e}")
        
        if not text or text.strip() == "":
            text = "not_verified"
        
        # Clean up the extracted text
        text = ' '.join(text.split())  # Remove extra whitespace
        
        fields[class_name] = text
        print(f"✅ {class_name}: '{text}'")

        # Draw boxes and label
        cv2.rectangle(image_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_drawn, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return fields, image_drawn


def extract_bonafide_fields(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')

    fields = {
        "college_name": None,
        "student_name": None,
        "class": None,
        "academic_year": None
    }

    college_match = re.search(r'(?i)([A-Z ]+LAW COLLEGE)', text)
    if college_match:
        fields['college_name'] = college_match.group(1).title().strip()

    name_match = re.search(r'This is to certify that\s+(?:KU\.?\s+)?([A-Z ]+?)\s+is/was', text)
    if name_match:
        fields['student_name'] = name_match.group(1).title().strip()

    class_match = re.search(r'class\s+([A-Z\s]+\d+)', text)
    if class_match:
        fields['class'] = class_match.group(1).strip()

    year_match = re.search(r'academic year\s+([0-9]{4}-[0-9]{4})', text)
    if year_match:
        fields['academic_year'] = year_match.group(1).strip()

    return fields


def extract_caste_fields(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    text = re.sub(r"\s+", " ", text)

    fields = {
        "applicant_name": None,
        "caste": None,
        "caste_category": None
    }

    # 👤 Name after "This is to certify that"
    name_match = re.search(r"this is to certify that\s+(.*?)\s+belongs to", text, re.IGNORECASE)

    # 🏷️ Caste between "belongs to" and "caste"
    caste_match = re.search(r"belongs to\s+(.*?)\s+caste", text, re.IGNORECASE)

    # 🗂️ Caste category after "which is recognized as"
    category_match = re.search(
        r"which is recognized as\s+(Scheduled Caste|Scheduled Tribe|Other Backward Class|Backward Class|General|OBC|SC|ST)",
        text,
        re.IGNORECASE
    )

    if name_match:
        fields["applicant_name"] = name_match.group(1).strip()

    if caste_match:
        fields["caste"] = caste_match.group(1).strip()

    if category_match:
        fields["caste_category"] = category_match.group(1).strip().title()

    return fields


def extract_marksheet_fields(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')

    fields = {
        "student_name": None,
        "roll_number": None,
        "percentage": None
    }

    keyword_map = {
        "name of the student": "student_name",
        "roll no": "roll_number",
        "percentage": "percentage"
    }

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    for line in lines:
        for keyword, field in keyword_map.items():
            if keyword in line:
                # Get the part after the keyword and cleanup
                try:
                    after = line.split(keyword)[1]
                    after = after.strip(" :.-").strip()
                    fields[field] = after
                except:
                    fields[field] = "not_found"

    return fields


def image_to_base64(image):
    """Convert OpenCV image to base64 string for JSON response"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str


# Flask Routes
@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Document Verification System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                padding: 30px;
                text-align: center;
                color: white;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .form-container {
                padding: 40px;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
                font-size: 1.1em;
            }
            
            input[type="text"], input[type="file"], select {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s ease;
            }
            
            input[type="text"]:focus, input[type="file"]:focus, select:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .upload-area {
                border: 2px dashed #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background: #f9f9f9;
                transition: all 0.3s ease;
                cursor: pointer;
                margin-bottom: 20px;
            }
            
            .upload-area:hover {
                border-color: #4CAF50;
                background: #f0f8f0;
            }
            
            .upload-area.dragover {
                border-color: #4CAF50;
                background: #e8f5e8;
            }
            
            .upload-icon {
                font-size: 3em;
                color: #ddd;
                margin-bottom: 15px;
            }
            
            .document-list {
                background: #f5f5f5;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .document-item {
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border: 2px solid #e0e0e0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .document-info {
                flex: 1;
            }
            
            .document-name {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            
            .document-controls {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .document-controls select {
                width: 200px;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            
            .remove-btn {
                background: #f44336;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
            }
            
            .remove-btn:hover {
                background: #d32f2f;
            }
            
            .submit-btn {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                width: 100%;
            }
            
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
            }
            
            .submit-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4CAF50;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result {
                margin-top: 30px;
                padding: 30px;
                border: 2px solid #ddd;
                border-radius: 10px;
                background: #f9f9f9;
                display: none;
            }
            
            .result.success {
                border-color: #4CAF50;
                background: #f1f8e9;
            }
            
            .result.error {
                border-color: #f44336;
                background: #ffebee;
            }
            
            .verification-summary {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .document-results {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .document-result {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .confidence-meter {
                display: flex;
                align-items: center;
                gap: 15px;
                margin: 15px 0;
            }
            
            .speedometer {
                width: 80px;
                height: 40px;
                position: relative;
                border-radius: 80px 80px 0 0;
                background: linear-gradient(to right, #ff4444 0%, #ffaa00 50%, #44ff44 100%);
                overflow: hidden;
            }
            
            .speedometer::before {
                content: '';
                position: absolute;
                top: 10px;
                left: 10px;
                right: 10px;
                bottom: 0;
                background: white;
                border-radius: 60px 60px 0 0;
            }
            
            .needle {
                position: absolute;
                bottom: 0;
                left: 50%;
                width: 2px;
                height: 30px;
                background: #333;
                transform-origin: bottom center;
                z-index: 10;
                transition: transform 0.5s ease;
            }
            
            .needle::after {
                content: '';
                position: absolute;
                bottom: -5px;
                left: -3px;
                width: 8px;
                height: 8px;
                background: #333;
                border-radius: 50%;
            }
            
            .confidence-text {
                font-size: 1.2em;
                font-weight: 600;
                color: #333;
            }
            
            .fields-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .field-item {
                background: #f5f5f5;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
            }
            
            .field-label {
                font-weight: 600;
                color: #4CAF50;
                text-transform: uppercase;
                font-size: 0.8em;
                margin-bottom: 5px;
            }
            
            .field-value {
                font-size: 1em;
                color: #333;
                word-wrap: break-word;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                margin-left: 10px;
            }
            
            .status-success {
                background: #e8f5e9;
                color: #2e7d32;
            }
            
            .status-warning {
                background: #fff3e0;
                color: #ef6c00;
            }
            
            .status-error {
                background: #ffebee;
                color: #c62828;
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .form-container {
                    padding: 20px;
                }
                
                .document-results {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 AI Document Verification System</h1>
                <p>Verify your identity by uploading multiple documents</p>
            </div>
            
            <div class="form-container">
                <form id="verifyForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="applicant_name">👤 Enter Your Application Name:</label>
                        <input type="text" id="applicant_name" name="applicant_name" 
                               placeholder="Enter your full name as per application" required>
                    </div>
                    
                    <div class="form-group">
                        <label>📄 Upload Documents:</label>
                        <div class="upload-area" id="uploadArea">
                            <div class="upload-icon">📁</div>
                            <h3>Drag & Drop Multiple Documents Here</h3>
                            <p>or click to browse and select multiple files</p>
                            <br>
                            <small>Supported: Aadhaar Card, Election Card, Ration Card<br>
                            Max size: 10MB per file | Formats: JPG, PNG, PDF<br>
                            <em>You can upload multiple documents at once</em></small>
                        </div>
                        <input type="file" id="fileInput" name="files" multiple 
                               accept="image/*,.pdf" style="display: none;">
                    </div>
                    
                    <div class="document-list" id="documentList" style="display: none;">
                        <h3>📋 Configure Your Documents:</h3>
                        <div id="documentItems"></div>
                    </div>
                    
                    <button type="submit" class="submit-btn" id="submitBtn" disabled>
                        🔍 Verify All Documents
                    </button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your documents...</p>
                </div>
                
                <div class="result" id="result">
                    <!-- Results will be populated here -->
                </div>
            </div>
        </div>
        
        <script>
            let selectedFiles = [];
            let documentTypes = {};
            
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const documentList = document.getElementById('documentList');
            const documentItems = document.getElementById('documentItems');
            const submitBtn = document.getElementById('submitBtn');
            
            // Document type options
            const docTypes = [
                'Aadhar Card',
                'PAN Card',
                'Handicap Smart Card',
                'Birth Certificate',
                'Bonafide Certificate',
                'Caste Certificate',
                'Salary Slip',
                'Passport',
                'Marksheet',
                'Transgender Certificate'
            ];
            
            // Upload area events
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });
            
            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });
            
            function handleFiles(files) {
                for (let file of files) {
                    if (file.size > 10 * 1024 * 1024) {
                        alert(`File ${file.name} is too large. Maximum size is 10MB.`);
                        continue;
                    }
                    
                    if (!selectedFiles.find(f => f.name === file.name)) {
                        selectedFiles.push(file);
                        documentTypes[file.name] = '';
                    }
                }
                updateDocumentList();
            }
            
            function updateDocumentList() {
                if (selectedFiles.length === 0) {
                    documentList.style.display = 'none';
                    submitBtn.disabled = true;
                    return;
                }
                
                documentList.style.display = 'block';
                documentItems.innerHTML = '';
                
                selectedFiles.forEach((file, index) => {
                    const div = document.createElement('div');
                    div.className = 'document-item';
                    div.innerHTML = `
                        <div class="document-info">
                            <div class="document-name">${file.name}</div>
                            <small>${(file.size / 1024 / 1024).toFixed(2)} MB</small>
                        </div>
                        <div class="document-controls">
                            <select onchange="updateDocumentType('${file.name}', this.value)">
                                <option value="">Select Document Type</option>
                                ${docTypes.map(type => 
                                    `<option value="${type}" ${documentTypes[file.name] === type ? 'selected' : ''}>${type}</option>`
                                ).join('')}
                            </select>
                            <button type="button" class="remove-btn" onclick="removeDocument(${index})">Remove</button>
                        </div>
                    `;
                    documentItems.appendChild(div);
                });
                
                // Check if all documents have types selected
                const allTypesSelected = selectedFiles.every(file => documentTypes[file.name] !== '');
                submitBtn.disabled = !allTypesSelected;
            }
            
            function updateDocumentType(fileName, type) {
                documentTypes[fileName] = type;
                updateDocumentList();
            }
            
            function removeDocument(index) {
                const fileName = selectedFiles[index].name;
                selectedFiles.splice(index, 1);
                delete documentTypes[fileName];
                updateDocumentList();
            }
            
            // Form submission
            document.getElementById('verifyForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const applicantName = document.getElementById('applicant_name').value;
                
                formData.append('applicant_name', applicantName);
                
                selectedFiles.forEach(file => {
                    formData.append('files', file);
                });
                
                formData.append('document_types', JSON.stringify(documentTypes));
                
                const loadingDiv = document.getElementById('loading');
                const resultDiv = document.getElementById('result');
                
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/verify_multiple', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        displayResults(data);
                    } else {
                        displayError(data.error || 'An error occurred');
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    displayError('Network error: ' + error.message);
                }
            });
            
            function displayResults(data) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result success';
                
                let html = `
                    <h2>✅ Document Verification Results</h2>
                    <div class="verification-summary">
                        <h3>📊 Verification Summary:</h3>
                        <p><strong>Application Name:</strong> ${data.applicant_name}</p>
                        <p><strong>Total Documents Processed:</strong> ${data.total_documents}</p>
                        <p><strong>Successful Matches:</strong> ${data.successful_matches}</p>
                        <p><strong>Overall Confidence:</strong> ${data.overall_confidence}%</p>
                        <p><strong>Processing Time:</strong> ${data.processing_time} seconds</p>
                    </div>
                    
                    <h3>📋 Individual Document Results:</h3>
                    <div class="document-results">
                `;
                
                data.results.forEach(result => {
                    const confidence = result.confidence || 0;
                    const statusClass = confidence >= 80 ? 'status-success' : 
                                       confidence >= 60 ? 'status-warning' : 'status-error';
                    const statusText = confidence >= 80 ? 'Verified' : 
                                      confidence >= 60 ? 'Partial Match' : 'Not Verified';
                    
                    html += `
                        <div class="document-result">
                            <h4>${result.document_type}<span class="status-badge ${statusClass}">${statusText}</span></h4>
                            <div class="confidence-meter">
                                <div class="speedometer">
                                    <div class="needle" style="transform: rotate(${(confidence / 100) * 180 - 90}deg)"></div>
                                </div>
                                <div class="confidence-text">${confidence}% confidence</div>
                            </div>
                            <div class="fields-container">
                    `;
                    
                    Object.entries(result.fields).forEach(([key, value]) => {
                        html += `
                            <div class="field-item">
                                <div class="field-label">${key.replace(/_/g, ' ')}</div>
                                <div class="field-value">${value || 'Not detected'}</div>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                
                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
                
                // Animate needles
                setTimeout(() => {
                    document.querySelectorAll('.needle').forEach(needle => {
                        needle.style.transition = 'transform 1s ease-out';
                    });
                }, 100);
            }
            
            function displayError(message) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `
                    <h2>❌ Error</h2>
                    <p style="color: #f44336; font-weight: 600; font-size: 1.1em;">${message}</p>
                `;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''
    return html_template


@app.route('/verify', methods=['POST'])
def verify_document():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        doc_type = request.form.get('doc_type')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not doc_type:
            return jsonify({'error': 'Document type not selected'}), 400
        
        # Save uploaded file
        file_id = uuid.uuid4().hex[:8]
        file_extension = file.filename.split('.')[-1].lower()
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
        file.save(file_path)
        
        # Load and process image
        image = load_image(file_path)
        model_path = DOC_MODEL_PATHS[doc_type]
        
        result = {
            'document_type': doc_type,
            'file_id': file_id,
            'fields': {},
            'processing_method': '',
            'annotated_image': None
        }
        
        # Process based on document type
        if doc_type == "Bonafide Certificate":
            fields = extract_bonafide_fields(image)
            result['fields'] = fields
            result['processing_method'] = 'Regex - Bonafide'
            
        elif doc_type == "Caste Certificate":
            fields = extract_caste_fields(image)
            result['fields'] = fields
            result['processing_method'] = 'Regex - Caste'
            
        elif doc_type == "Marksheet":
            fields = extract_marksheet_fields(image)
            result['fields'] = fields
            result['processing_method'] = 'Keyword-based - Marksheet'
        else:
            # YOLO + OCR processing
            if not model_path or not os.path.exists(model_path):
                return jsonify({'error': f'Model not found for {doc_type}'}), 400
            
            fields, annotated_image = run_yolo_ocr(image, model_path)
            result['fields'] = fields
            result['processing_method'] = 'YOLO + Enhanced OCR'
            result['annotated_image'] = image_to_base64(annotated_image)
            
            # Save annotated image
            output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{file_id}.jpg")
            cv2.imwrite(output_path, annotated_image)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/verify_multiple', methods=['POST'])
def verify_multiple_documents():
    try:
        # Get applicant name
        applicant_name = request.form.get('applicant_name')
        if not applicant_name:
            return jsonify({'error': 'Applicant name is required'}), 400
        
        # Get files
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        # Get document types
        document_types = json.loads(request.form.get('document_types', '{}'))
        
        results = []
        successful_matches = 0
        total_confidence = 0
        processing_start = time.time()
        
        for file in files:
            if file.filename == '':
                continue
                
            try:
                # Save uploaded file
                file_id = uuid.uuid4().hex[:8]
                file_extension = file.filename.split('.')[-1].lower()
                file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
                file.save(file_path)
                
                # Get document type
                doc_type = document_types.get(file.filename, '')
                if not doc_type:
                    continue
                
                # Load and process image
                image = load_image(file_path)
                model_path = DOC_MODEL_PATHS.get(doc_type)
                
                # Process document
                fields = {}
                confidence = 0
                
                if doc_type == "Bonafide Certificate":
                    fields = extract_bonafide_fields(image)
                    confidence = calculate_bonafide_confidence(fields, applicant_name)
                    
                elif doc_type == "Caste Certificate":
                    fields = extract_caste_fields(image)
                    confidence = calculate_caste_confidence(fields, applicant_name)
                    
                elif doc_type == "Marksheet":
                    fields = extract_marksheet_fields(image)
                    confidence = calculate_marksheet_confidence(fields, applicant_name)
                    
                else:
                    # YOLO + OCR processing
                    if model_path and os.path.exists(model_path):
                        fields, _ = run_yolo_ocr(image, model_path)
                        confidence = calculate_yolo_confidence(fields, applicant_name, doc_type)
                
                # Count successful matches
                if confidence >= 60:
                    successful_matches += 1
                
                total_confidence += confidence
                
                results.append({
                    'filename': file.filename,
                    'document_type': doc_type,
                    'fields': fields,
                    'confidence': round(confidence, 2),
                    'processing_method': get_processing_method(doc_type)
                })
                
                # Clean up uploaded file
                os.remove(file_path)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'document_type': document_types.get(file.filename, 'Unknown'),
                    'fields': {},
                    'confidence': 0,
                    'error': str(e)
                })
        
        processing_time = round(time.time() - processing_start, 2)
        overall_confidence = round(total_confidence / len(results), 2) if results else 0
        
        return jsonify({
            'applicant_name': applicant_name,
            'total_documents': len(results),
            'successful_matches': successful_matches,
            'overall_confidence': overall_confidence,
            'processing_time': processing_time,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Helper functions for confidence calculation
def calculate_bonafide_confidence(fields, applicant_name):
    confidence = 0
    
    # Name matching
    if fields.get('student_name'):
        if name_similarity(fields['student_name'], applicant_name) > 0.8:
            confidence += 40
        elif name_similarity(fields['student_name'], applicant_name) > 0.6:
            confidence += 25
    
    # Other fields present
    if fields.get('college_name'):
        confidence += 20
    if fields.get('class'):
        confidence += 20
    if fields.get('academic_year'):
        confidence += 20
    
    return min(confidence, 100)


def calculate_caste_confidence(fields, applicant_name):
    confidence = 0
    
    # Name matching
    if fields.get('applicant_name'):
        if name_similarity(fields['applicant_name'], applicant_name) > 0.8:
            confidence += 50
        elif name_similarity(fields['applicant_name'], applicant_name) > 0.6:
            confidence += 30
    
    # Other fields present
    if fields.get('caste'):
        confidence += 25
    if fields.get('caste_category'):
        confidence += 25
    
    return min(confidence, 100)


def calculate_marksheet_confidence(fields, applicant_name):
    confidence = 0
    
    # Name matching
    if fields.get('student_name'):
        if name_similarity(fields['student_name'], applicant_name) > 0.8:
            confidence += 50
        elif name_similarity(fields['student_name'], applicant_name) > 0.6:
            confidence += 30
    
    # Other fields present
    if fields.get('roll_number'):
        confidence += 25
    if fields.get('percentage'):
        confidence += 25
    
    return min(confidence, 100)


def calculate_yolo_confidence(fields, applicant_name, doc_type):
    confidence = 0
    
    # Name matching logic based on document type
    name_field = None
    if doc_type == "Aadhar Card":
        name_field = fields.get('name')
    elif doc_type == "PAN Card":
        name_field = fields.get('name')
    elif doc_type == "Passport":
        name_field = fields.get('name')
    
    if name_field and name_field != "not_verified":
        similarity = name_similarity(name_field, applicant_name)
        if similarity > 0.8:
            confidence += 50
        elif similarity > 0.6:
            confidence += 30
        elif similarity > 0.4:
            confidence += 15
    
    # Add confidence based on other verified fields
    verified_fields = sum(1 for value in fields.values() if value and value != "not_verified")
    total_fields = len(fields)
    
    if total_fields > 0:
        field_confidence = (verified_fields / total_fields) * 50
        confidence += field_confidence
    
    return min(confidence, 100)


def name_similarity(name1, name2):
    """Calculate similarity between two names using multiple methods"""
    if not name1 or not name2:
        return 0
    
    # Normalize names
    name1 = name1.lower().strip()
    name2 = name2.lower().strip()
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Split into words and find common words
    words1 = set(name1.split())
    words2 = set(name2.split())
    
    if not words1 or not words2:
        return 0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    jaccard_similarity = intersection / union if union > 0 else 0
    
    # Calculate word-level similarity
    word_similarities = []
    for word1 in words1:
        max_sim = 0
        for word2 in words2:
            # Simple character-based similarity
            if len(word1) >= 3 and len(word2) >= 3:
                if word1 in word2 or word2 in word1:
                    max_sim = max(max_sim, 0.8)
                else:
                    # Check character overlap
                    common_chars = set(word1).intersection(set(word2))
                    char_sim = len(common_chars) / max(len(word1), len(word2))
                    max_sim = max(max_sim, char_sim)
        word_similarities.append(max_sim)
    
    avg_word_similarity = sum(word_similarities) / len(word_similarities) if word_similarities else 0
    
    # Combine similarities
    final_similarity = (jaccard_similarity * 0.7) + (avg_word_similarity * 0.3)
    
    return final_similarity


def get_processing_method(doc_type):
    """Get processing method description"""
    if doc_type == "Bonafide Certificate":
        return "Regex Pattern Matching"
    elif doc_type == "Caste Certificate":
        return "Regex Pattern Matching"
    elif doc_type == "Marksheet":
        return "Keyword-based Extraction"
    else:
        return "YOLO + Enhanced OCR"

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Document Verification API'})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)