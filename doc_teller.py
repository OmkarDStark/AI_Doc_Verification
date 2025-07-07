import easyocr
import cv2
import numpy as np
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_field_in_ocr(image_path, query_field, similarity_threshold=0.6, languages=['en', 'hi', 'mr']):
    # Step 1: OCR
    reader = easyocr.Reader(languages)
    result = reader.readtext(image_path)

    found_results = []

    for i, (bbox, text, conf) in enumerate(result):
        if similar(query_field, text) > similarity_threshold:
            context = []

            # Previous 2 text blocks
            for j in range(max(0, i - 2), i):
                context.append(result[j][1])
            
            # Matched text
            context.append(text)

            # Next 2 text blocks
            for j in range(i + 1, min(i + 3, len(result))):
                context.append(result[j][1])
            
            found_results.append({
                "match": text,
                "context": context,
                "confidence": conf,
                "bounding_box": bbox
            })

    return found_results

# 🧪 Example usage:
if __name__ == "__main__":
    image_path = "/home/stark/Desktop/Coding/AI-Doc-Verification/Docs/passbook_page1.jpg"  # update with your path
    query_field = input("Enter field to search (e.g., account number, खाते क्रमांक): ")
    
    matches = find_field_in_ocr(image_path, query_field)

    if matches:
        print(f"\n🔍 Found {len(matches)} match(es) for '{query_field}':\n")
        for m in matches:
            print(f"Matched Text     : {m['match']}")
            print(f"Confidence       : {m['confidence']:.2f}")
            print(f"Bounding Box     : {m['bounding_box']}")
            print(f"Context Around   : {m['context']}\n")
    else:
        print(f"❌ No similar fields found for '{query_field}'.")
