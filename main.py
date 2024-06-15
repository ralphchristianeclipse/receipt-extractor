import csv
from transformers import pipeline
from PIL import Image
import glob
import os
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\MV\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Initialize the pipeline with GPU support
pipe = pipeline(
    "document-question-answering",
    model="naver-clova-ix/donut-base-finetuned-docvqa",
    device=0  # Specify GPU device (0 for first GPU, 1 for second GPU, etc.)
)

# Define the path to the folder containing images
folder_path = "receipts"  # Replace with your folder path

# Use glob to get a list of all image files in the folder
image_paths = glob.glob(os.path.join(folder_path, "*.[pj][pn]g"))

csv_file = "receipts_results.csv"

# Check if CSV file exists; if not, create an empty DataFrame
if os.path.exists(csv_file):
    csv = pd.read_csv(csv_file)
else:
    csv = pd.DataFrame(columns=["image_path", "total", "receipt_date"])

last_length = len(csv)  # Get the initial length of the DataFrame

# Function to process each image
def process_image(image_path):
    if image_path in csv["image_path"].values:
        print(f"{image_path} already processed. Skipping.\n")
        return None
    print(f"{image_path} Processing...\n")
    image = Image.open(image_path)
    total = pipe(image, "What is the total purchase?")
    receipt_date = pipe(image, "What is the receipt date?")
    return {
        "image_path": image_path,
        "total": total[0]['answer'],
        "receipt_date": receipt_date[0]['answer']
    }

# Process images in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers as per your GPU capacity
    futures = {executor.submit(process_image, image_path): image_path for image_path in image_paths}

    for future in as_completed(futures):
        result = future.result()
        if result:
            # Append the result to the DataFrame
            csv = pd.concat([csv,pd.DataFrame([result])])

# Check if any new rows were added
if len(csv) > last_length:
    # Save DataFrame to CSV file
    csv.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
else:
    print("All processed images were already in the CSV or skipped.")
