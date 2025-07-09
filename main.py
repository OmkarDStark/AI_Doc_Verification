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
    "Caste Certificate": "/home/stark/Desktop/Document Verification/models/castelast.pt",
    "Salary Slip": "/home/stark/Desktop/Document Verification/models/salarysliplast.pt",
    "Passport": "/home/stark/Desktop/Document Verification/models/passport.pt",
    "Marksheet": None
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
        <title>Document Verification System</title>
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
                max-width: 900px;
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
            
            input[type="file"], select {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s ease;
            }
            
            input[type="file"]:focus, select:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            select {
                background-color: white;
                cursor: pointer;
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
            
            .submit-btn:active {
                transform: translateY(0);
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
            
            .result h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            .fields-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .field-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .field-label {
                font-weight: 600;
                color: #4CAF50;
                text-transform: uppercase;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            
            .field-value {
                font-size: 1.1em;
                color: #333;
                word-wrap: break-word;
            }
            
            .image-container {
                text-align: center;
                margin-top: 30px;
            }
            
            .annotated-image {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                margin-top: 15px;
            }
            
            .processing-method {
                background: #e3f2fd;
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-weight: 600;
                color: #1976d2;
            }
            
            .error-message {
                color: #f44336;
                font-weight: 600;
                font-size: 1.1em;
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
                
                .fields-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 Document Verification System</h1>
                <p>Upload and verify your documents with AI-powered OCR technology</p>
            </div>
            
            <div class="form-container">
                <form id="verifyForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">📁 Upload Document (Image or PDF):</label>
                        <input type="file" id="file" name="file" accept="image/*,.pdf" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="doc_type">📋 Select Document Type:</label>
                        <select id="doc_type" name="doc_type" required>
                            <option value="">Choose document type...</option>
                            <option value="Aadhar Card">Aadhar Card</option>
                            <option value="PAN Card">PAN Card</option>
                            <option value="Handicap Smart Card">Handicap Smart Card</option>
                            <option value="Birth Certificate">Birth Certificate</option>
                            <option value="Bonafide Certificate">Bonafide Certificate</option>
                            <option value="Caste Certificate">Caste Certificate</option>
                            <option value="Salary Slip">Salary Slip</option>
                            <option value="Passport">Passport</option>
                            <option value="Marksheet">Marksheet</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="submit-btn">🔍 Verify Document</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your document...</p>
                </div>
                
                <div class="result" id="result">
                    <!-- Results will be populated here -->
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('verifyForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const loadingDiv = document.getElementById('loading');
                const resultDiv = document.getElementById('result');
                
                // Show loading, hide results
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/verify', {
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
                    <div class="processing-method">
                        🔧 Processing Method: ${data.processing_method}
                    </div>
                    <div class="fields-container">
                `;
                
                // Display extracted fields
                for (const [key, value] of Object.entries(data.fields)) {
                    html += `
                        <div class="field-item">
                            <div class="field-label">${key.replace(/_/g, ' ')}</div>
                            <div class="field-value">${value || 'Not detected'}</div>
                        </div>
                    `;
                }
                
                html += '</div>';
                
                // Display annotated image if available
                if (data.annotated_image) {
                    html += `
                        <div class="image-container">
                            <h3>📸 Annotated Image</h3>
                            <img src="data:image/jpeg;base64,${data.annotated_image}" 
                                 alt="Annotated Document" 
                                 class="annotated-image">
                        </div>
                    `;
                }
                
                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
            }
            
            function displayError(message) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `
                    <h2>❌ Error</h2>
                    <p class="error-message">${message}</p>
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


@app.route('/api/verify', methods=['POST'])
def api_verify():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data or 'doc_type' not in data:
            return jsonify({'error': 'Missing required fields: image_base64, doc_type'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        doc_type = data['doc_type']
        model_path = DOC_MODEL_PATHS.get(doc_type)
        
        if doc_type not in DOC_MODEL_PATHS:
            return jsonify({'error': f'Unsupported document type: {doc_type}'}), 400
        
        result = {
            'document_type': doc_type,
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
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Document Verification API'})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)