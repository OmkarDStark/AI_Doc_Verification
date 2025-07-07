import cv2
import json
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os
import string  # Add this at top if not already


fake = Faker('en_IN')

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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
        return fake.name()
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
        if subtype == "address":
            return fake.address().replace("\n", ", ")
        elif subtype == "paragraph":
            return fake.paragraph()
        elif subtype == "sentence":
            return fake.sentence()
        else:
            return fake.text()
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
        scale_factor = max_dim / max(img_height, img_width)
        resized = cv2.resize(img_original, (int(img_width * scale_factor), int(img_height * scale_factor)))
        cv2.imwrite(image_path, resized)  # overwrite original
        img_original = resized
        print(f"📏 Image resized to {resized.shape[1]}x{resized.shape[0]} and saved over original.")

    # Start at 1.0 zoom level
    zoom_level = 1.0
    img_base = img_original.copy()
    img_display = img_base.copy()
    img_copy = img_display.copy()

    def update_display():
        w = int(img_base.shape[1] * zoom_level)
        h = int(img_base.shape[0] * zoom_level)
        return cv2.resize(img_base, (w, h))

    cv2.namedWindow("Draw Fields", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Fields", mouse_callback)

    field_count = 0
    while True:
        print(f"\n🖌️ Draw field #{field_count + 1} (drag on image)...")
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
                print(f"🔍 Zoomed in: {zoom_level:.1f}x")
            elif key == ord('-'):
                zoom_level = max(zoom_level - 0.1, 0.3)
                print(f"🔎 Zoomed out: {zoom_level:.1f}x")

            if not drawing and current_rect:
                break

        # Adjust box to original coordinates
        x1, y1 = current_rect[0]
        x2, y2 = current_rect[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Map from display (zoomed) to original
        orig_x1 = int(x1 / zoom_level)
        orig_y1 = int(y1 / zoom_level)
        orig_x2 = int(x2 / zoom_level)
        orig_y2 = int(y2 / zoom_level)

        print(f"Box drawn (original scale): ({orig_x1}, {orig_y1}) → ({orig_x2}, {orig_y2})")
        confirm = input("✅ Keep this box? (y/n): ").strip().lower()
        if confirm != 'y':
            print("↩️ Redraw...")
            continue

        field_name = input("📝 Field label (e.g. name, dob, etc.): ").strip()
        while True:
            field_type = input("Select field type [name/date/number/list/text/alphanumeric]: ").strip().lower()
            if field_type in ["name", "date", "number", "list", "text", "alphanumeric"]:
                break
            else:
                print("❌ Invalid type. Choose from: name, date, number, list, text, alphanumeric")


        if field_type == "number":
            digit_count = input("How many digits? (default 6): ").strip()
            digit_count = int(digit_count) if digit_count.isdigit() else 6
            field_generators[field_name] = {"type": "number", "digits": digit_count}
        elif field_type == "alphanumeric":
            length = input("How many characters? (default 8): ").strip()
            length = int(length) if length.isdigit() else 8
            field_generators[field_name] = {"type": "alphanumeric", "length": length}
        elif field_type == "list":
            n = input("How many list entries? (e.g. 2): ").strip()
            n = int(n) if n.isdigit() else 2
            items = [input(f"Item {i+1}: ").strip() for i in range(n)]
            custom_lists[field_name] = items
            field_generators[field_name] = {"type": "list"}
        elif field_type == "text":
            subtype = input("Subtype [address/paragraph/sentence] (default: address): ").strip().lower()
            subtype = subtype if subtype in ["address", "paragraph", "sentence"] else "address"
            field_generators[field_name] = {"type": "text", "subtype": subtype}
        else:
            field_generators[field_name] = {"type": field_type}

        font_size = input(f"Font size (default {default_font_size}): ").strip()
        font_size = int(font_size) if font_size.isdigit() else default_font_size

        fields[field_name] = (orig_x1, orig_y1, orig_x2, orig_y2)
        field_font_sizes[field_name] = font_size

        # Draw final on base image for visual reference
        cv2.rectangle(img_base, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
        cv2.putText(img_base, field_name, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        field_count += 1

        cont = input("➕ Add another field? (y/n): ").strip().lower()
        if cont != 'y':
            print("✅ Annotation complete.")
            break

    cv2.destroyAllWindows()
    return fields, field_font_sizes, field_generators, custom_lists


def create_synthetic_doc(image_path, fields, font_sizes, generators, list_data, output_path="synthetic_output.jpg"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for label, (x1, y1, x2, y2) in fields.items():
        label_type = generators[label]["type"]
        value = generate_fake_data(label_type, label)
        font_size = font_sizes.get(label, default_font_size)
        font = ImageFont.truetype(font_path, font_size)

        draw.rectangle([x1, y1, x2, y2], fill="white")
        draw.text((x1 + 5, y1 + 2), value, font=font, fill="black")

    img.save(output_path)
    print(f"[✓] Saved: {output_path}")

# === Main ===
if __name__ == "__main__":
    img_path = input("📂 Enter path to document image: ").strip()

    save_fields = "fields.json"
    save_fonts = "field_font_sizes.json"
    save_generators = "field_generators.json"
    save_lists = "field_custom_lists.json"

    # Ask user if they want to define new fields
    reset = input("🔄 Do you want to define new fields for this template? (y/n): ").strip().lower()
    if reset == 'y':
        for f in [save_fields, save_fonts, save_generators, save_lists]:
            if os.path.exists(f):
                os.remove(f)
        print("🗑️ Old field definitions removed.")

    if all(os.path.exists(f) for f in [save_fields, save_fonts, save_generators]):
        print("[i] Found saved configuration.")
        with open(save_fields, "r") as f:
            fields = json.load(f)
        with open(save_fonts, "r") as f:
            field_font_sizes = json.load(f)
        with open(save_generators, "r") as f:
            field_generators = json.load(f)
        if os.path.exists(save_lists):
            with open(save_lists, "r") as f:
                custom_lists = json.load(f)
    else:
        fields, field_font_sizes, field_generators, custom_lists = draw_and_label_fields(img_path)
        with open(save_fields, "w") as f:
            json.dump(fields, f)
        with open(save_fonts, "w") as f:
            json.dump(field_font_sizes, f)
        with open(save_generators, "w") as f:
            json.dump(field_generators, f)
        with open(save_lists, "w") as f:
            json.dump(custom_lists, f)
        print("✅ Configuration saved.")

    num = input("📄 How many synthetic docs to generate? (default 5): ").strip()
    num = int(num) if num.isdigit() else 5

    prefix = input("📝 Enter output filename prefix (without extension, e.g. 'aadhaar_sample'): ").strip()
    prefix = prefix if prefix else "synthetic_doc"

    for i in range(num):
        out_path = f"{prefix}_{i+1}.jpg"
        create_synthetic_doc(img_path, fields, field_font_sizes, field_generators, custom_lists, output_path=out_path)
