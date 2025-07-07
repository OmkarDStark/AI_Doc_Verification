import cv2
import json
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os
import string
from indic_transliteration.sanscript import transliterate, HK, DEVANAGARI

# Faker & Fonts
fake = Faker('en_IN')
# english_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
english_font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
marathi_font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
default_font_size = 16

fields = {}
field_font_sizes = {}
field_generators = {}
custom_lists = {}

drawing = False
start_point = None
current_rect = None

def generate_fake_data(label_type, label_name=None):
    if label_type == "name":
        lang = field_generators[label_name].get("lang", "en")
        name_en = fake.name()
        name_mr = transliterate(name_en, HK, DEVANAGARI)
        return {"en": name_en, "mr": name_mr} if lang == "both" else name_en if lang == "en" else name_mr

    elif label_type == "date":
        return fake.date_of_birth().strftime("%d/%m/%Y")

    elif label_type == "number":
        digits = field_generators[label_name].get("digits", 6)
        return ''.join(str(random.randint(0, 9)) for _ in range(digits))

    elif label_type == "list":
        options = custom_lists.get(label_name, ["Option1", "Option2"])
        return random.choice(options)

    elif label_type == "text":
        subtype = field_generators[label_name].get("subtype", "address")
        lang = field_generators[label_name].get("lang", "en")
        addr_en = fake.address().replace("\n", ", ")
        addr_mr = transliterate(addr_en, HK, DEVANAGARI)
        if subtype == "paragraph":
            addr_en = fake.paragraph()
            addr_mr = transliterate(addr_en, HK, DEVANAGARI)
        elif subtype == "sentence":
            addr_en = fake.sentence()
            addr_mr = transliterate(addr_en, HK, DEVANAGARI)
        return {"en": addr_en, "mr": addr_mr} if lang == "both" else addr_en if lang == "en" else addr_mr

    elif label_type == "alphanumeric":
        length = field_generators[label_name].get("length", 8)
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=length))

    elif label_type == "email":
        return fake.email()

    elif label_type == "phone":
        return fake.phone_number()

    elif label_type == "url":
        return fake.url()

    else:
        return fake.word()

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (start_point, (x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (start_point, (x, y))

def draw_and_label_fields(image_path):
    global current_rect, drawing, start_point

    img_original = cv2.imread(image_path)
    img_height, img_width = img_original.shape[:2]

    # Auto-resize if needed
    max_dim = 1080
    if max(img_height, img_width) > max_dim:
        scale = max_dim / max(img_height, img_width)
        resized = cv2.resize(img_original, (int(img_width * scale), int(img_height * scale)))
        cv2.imwrite(image_path, resized)
        img_original = resized
        print(f"📏 Image resized to {resized.shape[1]}x{resized.shape[0]}")

    zoom_level = 1.0
    img_base = img_original.copy()

    def update_display():
        w = int(img_base.shape[1] * zoom_level)
        h = int(img_base.shape[0] * zoom_level)
        return cv2.resize(img_base, (w, h))

    cv2.namedWindow("Draw Fields", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Fields", mouse_callback)

    field_count = 0
    while True:
        print(f"\n🖌️ Draw field #{field_count + 1}")
        current_rect = None

        while True:
            temp_img = update_display()
            if current_rect:
                x1, y1 = current_rect[0]
                x2, y2 = current_rect[1]
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Draw Fields", temp_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('+'):
                zoom_level = min(zoom_level + 0.1, 3.0)
            elif key == ord('-'):
                zoom_level = max(zoom_level - 0.1, 0.3)
            if not drawing and current_rect:
                break

        x1, y1 = current_rect[0]
        x2, y2 = current_rect[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        ox1, oy1, ox2, oy2 = int(x1 / zoom_level), int(y1 / zoom_level), int(x2 / zoom_level), int(y2 / zoom_level)

        print(f"Box: ({ox1},{oy1}) → ({ox2},{oy2})")
        if input("✅ Keep? (y/n): ").strip().lower() != 'y':
            continue

        fname = input("📝 Field label: ").strip()
        while True:
            ftype = input("Type [name/date/number/list/text/alphanumeric]: ").strip().lower()
            if ftype in ["name", "date", "number", "list", "text", "alphanumeric"]:
                break

        if ftype == "number":
            digits = input("How many digits? (default 6): ").strip()
            field_generators[fname] = {"type": "number", "digits": int(digits) if digits.isdigit() else 6}
        elif ftype == "alphanumeric":
            length = input("Length? (default 8): ").strip()
            field_generators[fname] = {"type": "alphanumeric", "length": int(length) if length.isdigit() else 8}
        elif ftype == "list":
            count = int(input("How many options? (default 2): ").strip() or "2")
            items = [input(f"Item {i+1}: ").strip() for i in range(count)]
            custom_lists[fname] = items
            field_generators[fname] = {"type": "list"}
        elif ftype == "text":
            subtype = input("Subtype [address/paragraph/sentence] (default: address): ").strip().lower()
            subtype = subtype if subtype in ["address", "paragraph", "sentence"] else "address"
            lang = input("Language [en/mr/both] (default: en): ").strip().lower()
            if lang not in ["en", "mr", "both"]:
                lang = "en"
            field_generators[fname] = {"type": "text", "subtype": subtype, "lang": lang}
        elif ftype == "name":
            lang = input("Language [en/mr/both] (default: en): ").strip().lower()
            if lang not in ["en", "mr", "both"]:
                lang = "en"
            field_generators[fname] = {"type": "name", "lang": lang}
        else:
            field_generators[fname] = {"type": ftype}

        fsize = input(f"Font size (default {default_font_size}): ").strip()
        field_font_sizes[fname] = int(fsize) if fsize.isdigit() else default_font_size
        fields[fname] = (ox1, oy1, ox2, oy2)

        cv2.rectangle(img_base, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        cv2.putText(img_base, fname, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        field_count += 1

        if input("➕ Add more? (y/n): ").strip().lower() != 'y':
            break

    cv2.destroyAllWindows()
    return fields, field_font_sizes, field_generators, custom_lists

def create_synthetic_doc(image_path, fields, font_sizes, generators, list_data, output_path="synthetic_output.jpg"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for label, (x1, y1, x2, y2) in fields.items():
        ftype = generators[label]["type"]
        value = generate_fake_data(ftype, label)
        size = font_sizes.get(label, default_font_size)
        font_en = ImageFont.truetype(english_font_path, size)
        font_mr = ImageFont.truetype(marathi_font_path, size)

        draw.rectangle([x1, y1, x2, y2], fill="white")
        if isinstance(value, dict):
            y_offset = y1 + 2
            if "en" in value:
                draw.text((x1 + 5, y_offset), value["en"], font=font_en, fill="black")
                y_offset += size
            if "mr" in value:
                draw.text((x1 + 5, y_offset), value["mr"], font=font_mr, fill="black")
        else:
            draw.text((x1 + 5, y1 + 2), value, font=font_en, fill="black")

    img.save(output_path)
    print(f"[✓] Saved: {output_path}")

# === Main ===
if __name__ == "__main__":
    img_path = input("📂 Path to document image: ").strip()

    save_fields = "fields.json"
    save_fonts = "field_font_sizes.json"
    save_generators = "field_generators.json"
    save_lists = "field_custom_lists.json"

    reset = input("🔄 Define new fields for this template? (y/n): ").strip().lower()
    if reset == 'y':
        for f in [save_fields, save_fonts, save_generators, save_lists]:
            if os.path.exists(f): os.remove(f)

    if all(os.path.exists(f) for f in [save_fields, save_fonts, save_generators]):
        print("[i] Loaded previous configuration.")
        with open(save_fields) as f: fields = json.load(f)
        with open(save_fonts) as f: field_font_sizes = json.load(f)
        with open(save_generators) as f: field_generators = json.load(f)
        if os.path.exists(save_lists):
            with open(save_lists) as f: custom_lists = json.load(f)
    else:
        fields, field_font_sizes, field_generators, custom_lists = draw_and_label_fields(img_path)
        with open(save_fields, "w") as f: json.dump(fields, f)
        with open(save_fonts, "w") as f: json.dump(field_font_sizes, f)
        with open(save_generators, "w") as f: json.dump(field_generators, f)
        with open(save_lists, "w") as f: json.dump(custom_lists, f)
        print("✅ Field configuration saved.")

    count = input("📄 How many synthetic docs to generate? (default 5): ").strip()
    count = int(count) if count.isdigit() else 5
    prefix = input("📎 Output file prefix (default 'synthetic_doc'): ").strip() or "synthetic_doc"

    for i in range(count):
        output_file = f"{prefix}_{i+1}.jpg"
        create_synthetic_doc(img_path, fields, field_font_sizes, field_generators, custom_lists, output_file)
