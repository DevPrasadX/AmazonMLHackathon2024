import pandas as pd
import re
import os
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.constants import allowed_units  # Import allowed_units

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Directory to store downloaded images
download_dir = "downloaded_images"
os.makedirs(download_dir, exist_ok=True)

# Full name mapping for units
unit_full_name_map = {
    'cm': 'centimetre',
    'centimetre': 'centimetre',
    'ft': 'foot',
    'foot': 'foot',
    'in': 'inch',
    'inch': 'inch',
    'm': 'metre',
    'metre': 'metre',
    'mm': 'millimetre',
    'millimetre': 'millimetre',
    'yd': 'yard',
    'yard': 'yard',
    'g': 'gram',
    'gram': 'gram',
    'kg': 'kilogram',
    'kilogram': 'kilogram',
    'µg': 'microgram',
    'microgram': 'microgram',
    'mg': 'milligram',
    'milligram': 'milligram',
    'oz': 'ounce',
    'ounce': 'ounce',
    'lb': 'pound',
    'pound': 'pound',
    'ton': 'ton',
    'kV': 'kilovolt',
    'kilovolt': 'kilovolt',
    'mV': 'millivolt',
    'millivolt': 'millivolt',
    'V': 'volt',
    'volt': 'volt',
    'kW': 'kilowatt',
    'kilowatt': 'kilowatt',
    'W': 'watt',
    'watt': 'watt',
    'cl': 'centilitre',
    'centilitre': 'centilitre',
    'cu ft': 'cubic foot',
    'cubic foot': 'cubic foot',
    'cu in': 'cubic inch',
    'cubic inch': 'cubic inch',
    'cup': 'cup',
    'decilitre': 'decilitre',
    'dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'fluid ounce': 'fluid ounce',
    'gal': 'gallon',
    'gallon': 'gallon',
    'imp gal': 'imperial gallon',
    'imperial gallon': 'imperial gallon',
    'l': 'litre',
    'litre': 'litre',
    'µl': 'microlitre',
    'microlitre': 'microlitre',
    'ml': 'millilitre',
    'millilitre': 'millilitre',
    'pt': 'pint',
    'pint': 'pint',
    'qt': 'quart',
    'quart': 'quart'
}

# Preprocess image only when needed (based on image quality)
def preprocess_image(image, enhance=False):
    image = image.convert('L')
    if enhance:
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(2.0)
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(2.0)
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return image

# Function to download and save images locally
def download_image(image_url, index):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_path = os.path.join(download_dir, f'image_{index}.png')
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    except requests.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

# Function to extract text from an image file
def extract_text_from_image(image_path, enhance=False):
    try:
        image = Image.open(image_path)
        image = preprocess_image(image, enhance=enhance)
        config = "--psm 6"
        text = pytesseract.image_to_string(image, config=config)
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from {image_path}: {e}")
        return None

# Function to generate a regex pattern based on allowed units
def get_unit_pattern():
    unit_regex = r'\b(\d+(\.\d+)?)\s*({})\b'.format('|'.join(allowed_units))
    return re.compile(unit_regex, re.IGNORECASE)

# Function to extract the highest value unit from text based on entity's units
def extract_highest_unit(text):
    unit_pattern = get_unit_pattern()
    
    matches = unit_pattern.findall(text)
    if not matches:
        return None
    
    # Find the highest value
    max_value = max(float(match[0]) for match in matches)
    max_unit = unit_full_name_map.get(matches[0][2].lower(), matches[0][2].lower())
    
    return f"{max_value} {max_unit}"

# The predictor function integrated with image processing and text extraction
def predictor(image_link, category_id, entity_name):
    # Download the image
    image_path = download_image(image_link, category_id)
    if not image_path:
        return {"index": category_id, "entity_name": entity_name, "image_link": image_link, "units_extracted": None}
    
    # Extract text from the image
    text = extract_text_from_image(image_path, enhance=False)
    if text is None or text.strip() == '':
        text = extract_text_from_image(image_path, enhance=True)
    
    # Extract the highest value unit from the text
    highest_unit = extract_highest_unit(text)
    
    return {"index": category_id, "entity_name": entity_name, "image_link": image_link, "units_extracted": highest_unit}

if __name__ == "__main__":
    # Load the dataset from sample_test.csv
    csv_file_path = "sample_test.csv"
    df = pd.read_csv(csv_file_path)

    # Extract text and relevant units for each image using parallel processing
    results = []

    # ThreadPoolExecutor to process rows in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predictor, row['image_link'], row['index'], row['entity_name']) for _, row in df.iterrows()]

        # Use tqdm to show progress
        for future in tqdm(futures, total=len(futures), desc="Extracting data", unit="image"):
            results.append(future.result())

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a new CSV file
    output_filename = 'test_out.csv'
    df_results.to_csv(output_filename, index=False)
    
    print("Processing complete. Results saved to test_out.csv.")
