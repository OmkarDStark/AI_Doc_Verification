# pdf_to_images.py
# Converts all PDFs in a folder to JPGs (one per page), without deleting the originals.

from pdf2image import convert_from_path
from pathlib import Path

# 🔧 SET YOUR INPUT DIRECTORY HERE
input_dir = Path("PMC_Data/birthcertificates")  # e.g., Path("/home/user/Documents/my_docs")

def convert_pdfs_to_images(input_dir, dpi=200):
    for file in input_dir.iterdir():
        if file.suffix.lower() == ".pdf":
            print(f"📄 Converting PDF: {file.name}")
            try:
                images = convert_from_path(str(file), dpi=dpi)
                for i, img in enumerate(images):
                    img_path = file.with_name(f"{file.stem}_page{i+1}.jpg")
                    img.save(img_path, "JPEG")
                    print(f"✅ Saved: {img_path.name}")
            except Exception as e:
                print(f"❌ Failed to convert {file.name}: {e}")

if __name__ == "__main__":
    convert_pdfs_to_images(input_dir)
