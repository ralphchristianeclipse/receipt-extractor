import csv
from transformers import pipeline
from PIL import Image
import glob
import os
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import hashlib
import requests
from io import BytesIO
import re
from datetime import datetime

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
    csv = pd.DataFrame(columns=["image_path", "total", "receipt_date","receipt_issuer", "etag", "image_url"])

last_length = len(csv)  # Get the initial length of the DataFrame

def parse_date(date_str):
    if not date_str:
        return date_str

    # Define possible date formats
    date_formats = [
        "%B %d, %Y",    # 'December 22, 2014'
        "%B %d %Y",     # 'June 10 2044'
        "%d/%m/%Y",     # '26/08/2002'
        "%m/%d/%y",     # '06/14/29'
        "%B %d",        # 'June 14' (assuming current year)
    ]

    for fmt in date_formats:
        try:
            # Try parsing the date
            parsed_date = datetime.strptime(date_str, fmt)
            # Handle cases where only month and day are given (assume current year)
            if fmt == "%B %d":
                parsed_date = parsed_date.replace(year=datetime.now().year)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_str

def parse_float(value):
    # Use a regular expression to remove all non-numeric characters except for the period.
    cleaned_string = re.sub(r'[^\d.]', '', value)
    try:
        # Convert the cleaned string to a float
        return float(cleaned_string)
    except ValueError:
        # Return None or some default value if conversion fails
        return value


def upload_image(image: Image):

    # Save the image to a file-like object in memory
    image_file = BytesIO()
    image.save(image_file, format='JPEG', exif=image.info.get('exif'))
    image_file.seek(0)

    # Create a file object to be used with the requests library
    files = {'file': ('output.jpg', image_file, 'image/jpeg')}

    # Create the form data
    data = {
      'expiration': '10'  # expire in 3 minutes
    }
    # Make the POST request to upload the file
    response = requests.post('https://tmpfiles.org/api/v1/upload', files=files, data=data)

    # Check for success
    if response.status_code == 200:
        response_data = response.json()
        # Extract the URL and replace the domain if it exists
        if 'data' in response_data and 'url' in response_data['data']:
            original_url = response_data['data']['url']
            replaced_url = original_url.replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
            return replaced_url
        else:
            print('URL not found in the response')
            return None
    else:
        print('File upload failed')
        print('Response status code:', response.status_code)
        print('Response text:', response.text)
        return None

def calculate_image_etag(image):
    try:
        # Calculate MD5 hash of the image data
        md5_hash = hashlib.md5()

        # Convert image data to bytes
        image_bytes = image.tobytes()

        # Update hash with image data bytes
        md5_hash.update(image_bytes)

        # Compute hexadecimal digest of the hash
        etag = md5_hash.hexdigest()

        return etag

    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to process each image
def process_image(image_path):
    if image_path in csv["image_path"].values:
        print(f"{image_path} already processed. Skipping.\n")
        return None
    print(f"{image_path} Processing...\n")
    image = Image.open(image_path)
    etag = calculate_image_etag(image)
    if etag in csv["etag"].values:
      print(f"{image_path} ETag: ${etag} already processed. Skipping.\n")
      return None
    print(f"Uploading... {image_path}")
    image_url = upload_image(image)
    print(f"Uploaded... {image_path}")
    # TODO: make the three pipe functions run in parallel or batch them
    total = pipe(image, "What is the total?")
    receipt_date = pipe(image, "What is the receipt date?")
    receipt_issuer = pipe(image, "What is the receipt issuer?")
    return {
        "image_path": image_path,
        "total": parse_float(total[0]['answer']),
        "receipt_date": parse_date(receipt_date[0]['answer']),
        "receipt_issuer": receipt_issuer[0]['answer'],
        "etag": etag,
        "image_url": image_url
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
