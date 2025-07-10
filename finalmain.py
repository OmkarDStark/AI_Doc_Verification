from flask import Flask, request, jsonify, render_template
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
    "Current Month Salary Slip": "/home/stark/Desktop/Document Verification/models/salarysliplast.pt",
    "Passport": "/home/stark/Desktop/Document Verification/models/passport.pt",
    "Passport and VISA": "/home/stark/Desktop/Document Verification/models/passport.pt",
    "Marksheet": None
}

# Document classification keywords (from document_classifier.py)
DOCUMENT_KEYWORDS = {
    "Aadhar Card": ["uidai", "unique identification authority of india", "government of india", "aadhaar"],
    "Bank Account Pass Book": ["account no", "account number","ifsc", "branch", "bank", "branch code"],
    "Birth Certificate": ["name of child", "birth", "dob", "place of birth", "certificate of birth", "sex", "name of mother", "place of birth"],
    "Income Certificate": ["income", "certificate of income", "earning","वार्षिक उत्पन्न", "जिल्हा", "तलाठी अहवाल या आधारावर"],
    "Handicap Certificate": ["handicap", "disability", "percent disability", "physically challenged", "name of the hospital", "case of", "percentage", "unique disability id"],
    "Light bill": ["electricity bill", "consumer number", "mseb", "mahavitaran", "बिलींग युनिट", "पुरवठा दिनांक", "मंजुर भार", "चालु रिडिंग दिनांक"],
    "Telephone Bill Receipt": ["telephone", "bsnl", "billing date", "mobile number"],
    "Caste certificate": ["caste", "category", "scheduled caste", "scheduled tribe", "other backward class", "socially and educationally backward class", "caste validity", "sub divisional officer"],
    "Caste Certificate": ["caste", "category", "scheduled caste", "scheduled tribe", "other backward class", "socially and educationally backward class", "caste validity", "sub divisional officer"],
    "College fee receipt": ["fee", "tuition", "receipt", "college", "payment", "course", "class"],
    "Marksheet": ["marks", "subject", "grade", "exam", "percentage", "semester", "father's name", "school", "college", "sub code", "pr"],
    "Leaving Certificate": ["leaving", "conduct", "attendance", "school leaving","principal","student","date of admission","subject studied","reason for leaving","cbse","udise no","class teacher"],
    "Passport and VISA": ["passport", "visa", "republic of india", "expiry date","passport no","valid from","valid until","date of expiry","date of issue"],
    "Baseless certificate of Tehsildar / Talathi": ["tehsildar", "talathi", "baseless", "authority","रेशनकार्ड","विधवा","परीतवता" ,"घटस्फोटिता" ,"निराधार" ,"व्यक्तीला"],
    "Certificate of survival": ["survival", "alive", "living", "proof of survival","ह्यात"],
    "Death Certificate": ["death", "deceased", "passed away", "date of death","death certificate","age of deceased","place of death"],
    "Current Month Salary Slip": ["salary", "monthly pay", "employee code", "basic pay", "earnings","retirement date","basic pay","deduction","total salary","salary slip","basic","hra","hra","da","professional tax"],
    "Salary Slip": ["salary", "monthly pay", "employee code", "basic pay", "earnings","retirement date","basic pay","deduction","total salary","salary slip","basic","hra","hra","da","professional tax"],
    "NOC from PMC ward office": ["noc", "no objection", "pmc", "ward office","अर्ज"],
    "Marriage Certificate": ["marriage", "husband", "wife", "विवाह","विधी","विधी संपन्न"],
    "Handicap smart card": ["smart card", "disability", "handicap card", "govt issued","unique disability id","disability percentage","% of disability","ud id","disability type"],
    "Handicap Smart Card": ["smart card", "disability", "handicap card", "govt issued","unique disability id","disability percentage","% of disability","ud id","disability type"],
    "Transgender Certificate": ["transgender", "gender identity", "third gender","transgender identity card"],
    "Bonafide Certificate": ["bonafide", "student", "institution", "studying", "enrolled"],
    "PAN Card": ["pan", "permanent account number", "income tax department", "government of india", "pancard"]
}

def extract_text_ocr(image):
    """Extract text from image using OCR for document classification"""
    try:
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            else:
                image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Extract text using Tesseract with multiple languages
        text = pytesseract.image_to_string(image_pil, lang='mar+eng')
        return text.lower()
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return ""

def classify_document_type(image):
    """Classify document type based on extracted text and keywords"""
    try:
        # Extract text from image
        text = extract_text_ocr(image)
        
        if not text.strip():
            return "Unknown", 0
        
        # Calculate match scores for each document type
        scores = {}
        for doc_type, keywords in DOCUMENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                # Use regex to find whole word matches
                if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", text):
                    score += 1
            scores[doc_type] = score
        
        # Find the best match
        if scores:
            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]
            
            # Only return a classification if we have a reasonable confidence
            if best_score > 0:
                return best_match, best_score
        
        return "Unknown", 0
        
    except Exception as e:
        print(f"❌ Document classification failed: {e}")
        return "Unknown", 0

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
    return render_template('index.html')

@app.route('/verify_multiple', methods=['POST'])
def verify_multiple_documents():
    try:
        applicant_name = request.form.get('applicant_name')
        if not applicant_name:
            return jsonify({'error': 'Applicant name is required'}), 400
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        results = []
        successful_matches = 0
        total_confidence = 0
        import time
        processing_start = time.time()
        for file in files:
            if file.filename == '':
                continue
            try:
                file_id = uuid.uuid4().hex[:8]
                file_extension = file.filename.split('.')[-1].lower()
                file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
                file.save(file_path)
                image = load_image(file_path)
                doc_type, classification_score = classify_document_type(image)
                model_path = DOC_MODEL_PATHS.get(doc_type)
                fields = {}
                confidence = 0
                processing_method = ''
                annotated_image = None
                if doc_type == "Bonafide Certificate":
                    fields = extract_bonafide_fields(image)
                    confidence = calculate_bonafide_confidence(fields, applicant_name)
                    processing_method = 'Regex - Bonafide'
                elif doc_type == "Caste Certificate" or doc_type == "Caste certificate":
                    fields = extract_caste_fields(image)
                    confidence = calculate_caste_confidence(fields, applicant_name)
                    processing_method = 'Regex - Caste'
                elif doc_type == "Marksheet":
                    fields = extract_marksheet_fields(image)
                    confidence = calculate_marksheet_confidence(fields, applicant_name)
                    processing_method = 'Keyword-based - Marksheet'
                else:
                    if model_path and os.path.exists(model_path):
                        fields, annotated_image = run_yolo_ocr(image, model_path)
                        confidence = calculate_yolo_confidence(fields, applicant_name, doc_type)
                        processing_method = 'YOLO + Enhanced OCR'
                if confidence >= 60:
                    successful_matches += 1
                total_confidence += confidence
                results.append({
                    'filename': file.filename,
                    'document_type': doc_type,
                    'classification_score': classification_score,
                    'fields': fields,
                    'confidence': round(confidence, 2),
                    'processing_method': processing_method,
                    'annotated_image': image_to_base64(annotated_image) if annotated_image is not None else None
                })
                os.remove(file_path)
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'document_type': 'Unknown',
                    'classification_score': 0,
                    'fields': {},
                    'confidence': 0,
                    'processing_method': '',
                    'annotated_image': None,
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
        print(f"/verify_multiple error: {e}")
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