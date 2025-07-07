import cv2
import json
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os

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
    global current_rect
    img = cv2.imread(image_path)
    img_copy = img.copy()
    cv2.namedWindow("Draw Fields")
    cv2.setMouseCallback("Draw Fields", mouse_callback)

    field_count = 0
    while True:
        print(f"\n🖌️ Draw field #{field_count + 1} (click and drag box on image)...")
        current_rect = None

        while True:
            temp_img = img_copy.copy()
            if current_rect:
                cv2.rectangle(temp_img, current_rect[0], current_rect[1], (0, 255, 0), 2)
            cv2.imshow("Draw Fields", temp_img)
            key = cv2.waitKey(1)
            if not drawing and current_rect:
                break

        x1, y1 = current_rect[0]
        x2, y2 = current_rect[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        print(f"Box drawn: ({x1}, {y1}) → ({x2}, {y2})")
        confirm = input("✅ Keep this box? (y/n): ").strip().lower()
        if confirm != 'y':
            print("↩️ Redraw the box...")
            continue

        field_name = input("📝 Field label (e.g. name, dob, aadhaar): ").strip()

        while True:
            field_type = input("Select field type [name/date/number/list/text]: ").strip().lower()
            if field_type in ["name", "date", "number", "list", "text"]:
                break
            else:
                print("❌ Invalid type. Choose from: name, date, number, list, text")

        if field_type == "number":
            digit_count = input("How many digits? (default 6): ").strip()
            digit_count = int(digit_count) if digit_count.isdigit() else 6
            field_generators[field_name] = {"type": "number", "digits": digit_count}

        elif field_type == "list":
            n = input("How many list entries? (e.g. 2): ").strip()
            n = int(n) if n.isdigit() else 2
            items = []
            for i in range(n):
                item = input(f"Enter item {i+1}: ").strip()
                items.append(item)
            custom_lists[field_name] = items
            field_generators[field_name] = {"type": "list"}

        elif field_type == "text":
            print("📄 Available text subtypes: address, paragraph, sentence")
            subtype = input("Enter text subtype (default: address): ").strip().lower()
            subtype = subtype if subtype in ["address", "paragraph", "sentence"] else "address"
            field_generators[field_name] = {"type": "text", "subtype": subtype}

        else:
            field_generators[field_name] = {"type": field_type}

        font_size = input(f"Font size (default {default_font_size}): ").strip()
        font_size = int(font_size) if font_size.isdigit() else default_font_size

        fields[field_name] = (x1, y1, x2, y2)
        field_font_sizes[field_name] = font_size

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, field_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Draw Fields", img_copy)

        cont = input("➕ Add another field? (y/n): ").strip().lower()
        if cont != 'y':
            print("✅ Field annotation complete.")
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
    for i in range(num):
        out_path = f"synthetic_doc_{i+1}.jpg"
        create_synthetic_doc(img_path, fields, field_font_sizes, field_generators, custom_lists, output_path=out_path)
