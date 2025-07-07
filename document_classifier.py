# classify_documents_tesseract.py
# Fast OCR + keyword matching based classifier using Tesseract

from pathlib import Path
from PIL import Image
import pytesseract
import re

# 🔧 SET YOUR IMAGE DIRECTORY HERE
image_dir = Path("/home/stark/Desktop/Coding/AI-Doc-Verification/Docs")

# 21 classes + keywords per class
document_keywords = {
    "Aadhar Card": ["uidai", "unique identification authority of india", "government of india", "aadhaar"],
    "Bank Account Pass Book": ["account no", "account number","ifsc", "branch", "bank", "branch code"],
    "Birth Certificate": ["name of child", "birth", "dob", "place of birth", "certificate of birth", "sex", "name of mother", "place of birth"],
    "Income Certificate": ["income", "certificate of income", "earning","वार्षिक उत्पन्न", "जिल्हा", "तलाठी अहवाल या आधारावर"],
    "Handicap Certificate": ["handicap", "disability", "percent disability", "physically challenged", "name of the hospital", "case of", "percentage", "unique disability id"],
    "Light bill": ["electricity bill", "consumer number", "mseb", "mahavitaran", "बिलींग युनिट", "पुरवठा दिनांक", "मंजुर भार", "चालु रिडिंग दिनांक"],
    "Telephone Bill Receipt": ["telephone", "bsnl", "billing date", "mobile number"],
    "Caste certificate": ["caste", "category", "scheduled caste", "scheduled tribe", "other backward class", "socially and educationally backward class", "caste validity", "sub divisional officer"],
    "College fee receipt": ["fee", "tuition", "receipt", "college", "payment", "course", "class"],
    "Marksheet": ["marks", "subject", "grade", "exam", "percentage", "semester", "father's name", "school", "college", "sub code", "pr", ],
    "Leaving Certificate": ["leaving", "conduct", "attendance", "school leaving","principal","student","date of admission","subject studied","reason for leaving","cbse","udise no","class teacher"],
    "Passport and VISA": ["passport", "visa", "republic of india", "expiry date","passport no","valid from","valid until","date of expiry","date of issue"],
    "Baseless certificate of Tehsildar / Talathi": ["tehsildar", "talathi", "baseless", "authority","रेशनकार्ड","विधवा","परीतवता" ,"घटस्फोटिता" ,"निराधार" ,"व्यक्तीला"],
    "Certificate of survival": ["survival", "alive", "living", "proof of survival","ह्यात"],
    "Death Certificate": ["death", "deceased", "passed away", "date of death","death certificate","age of deceased","place of death"],
    "Current Month Salary Slip": ["salary", "monthly pay", "employee code", "basic pay", "earnings","retirement date","basic pay","deduction","total salary","salary slip","basic","hra","hra","da","professional tax"],
    "NOC from PMC ward office": ["noc", "no objection", "pmc", "ward office","अर्ज",],
    "Marriage Certificate": ["marriage", "husband", "wife", "विवाह","विधी","विधी संपन्न"],
    "Handicap smart card": ["smart card", "disability", "handicap card", "govt issued","unique disability id","disability percentage","% of disability","ud id","disability type"],
    "Transgender Certificate": ["transgender", "gender identity", "third gender","transgender identity card"],
    "Bonafide Certificate": ["bonafide", "student", "institution", "studying", "enrolled"]
}

def extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='mar+eng')
        return text.lower()
    except Exception as e:
        print(f"❌ OCR failed on {image_path.name}: {e}")
        return ""

def match_document_type(text):
    scores = {}
    for doc_type, keywords in document_keywords.items():
        score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text))
        scores[doc_type] = score
    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match]

def classify_documents(image_dir):
    for file in sorted(image_dir.iterdir()):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"\n🔍 Processing: {file.name}")
            try:
                text = extract_text(file)
                if not text.strip():
                    print("⚠️ No text detected.")
                    continue
                predicted_label, match_score = match_document_type(text)
                print(f"✅ Predicted Class: {predicted_label} (Matched {match_score} keywords)")
            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    classify_documents(image_dir)
