import os
import re
import requests
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from easyocr import Reader
from src.constants import allowed_units, unit_full_name_map, entity_unit_map

# Initialize EasyOCR Reader
reader = Reader(['en'])

# Directory to store downloaded images
download_dir = "downloaded_images"
os.makedirs(download_dir, exist_ok=True)

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
def download_image(image_url, group_id):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_path = os.path.join(download_dir, f'image_{group_id}.png')
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    except requests.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

# Function to extract text from an image file using EasyOCR
def extract_text_from_image(image_path, enhance=False):
    try:
        image = Image.open(image_path)
        image = preprocess_image(image, enhance=enhance)
        # Convert image to a format supported by EasyOCR
        image = image.convert('RGB')
        text_results = reader.readtext(image)
        # Extract and concatenate text from all detected regions
        text = ' '.join([result[1] for result in text_results])
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from {image_path}: {e}")
        return None

# Function to generate a regex pattern based on the entity's units
def get_unit_pattern(units):
    unit_regex = r'\b(\d+(\.\d+)?)\s*({})\b'.format('|'.join(units))
    return re.compile(unit_regex, re.IGNORECASE)

# Function to extract the highest value unit from text based on entity's units
def extract_highest_unit(entity_name, text):
    if entity_name not in entity_unit_map:
        return None

    # Get the set of valid units for the entity
    valid_units = entity_unit_map[entity_name]
    
    # Generate regex pattern for short form units
    unit_pattern = get_unit_pattern(unit_full_name_map.keys())
    
    # Find all matches of value + unit
    matches = unit_pattern.findall(text)
    if not matches:
        return None
    
    # Process each match: convert short unit to full form and check validity
    valid_matches = []
    for match in matches:
        value = float(match[0])
        short_unit = match[2].lower()
        
        # Convert short form to full form
        full_unit = unit_full_name_map.get(short_unit, short_unit)
        
        # Only keep valid units for the entity
        if full_unit in valid_units:
            valid_matches.append((value, full_unit))
    
    if not valid_matches:
        return None

    # Find the highest value
    max_value, max_unit = max(valid_matches, key=lambda x: x[0])

    # Filter the results based on allowed units
    if max_unit not in allowed_units:
        return None  # Skip if the max unit is not allowed

    # Format max_value: add decimal if float, leave as is if integer
    if max_value.is_integer():
        formatted_value = f"{int(max_value)}"  # No decimal for integers
    else:
        formatted_value = f"{max_value:.2f}"  # Two decimal places for floats

    return f"{formatted_value} {max_unit}"

# The predictor function integrated with image processing and text extraction
def predictor(image_link, group_id, entity_name, index):
    # Download the image
    image_path = download_image(image_link, group_id)
    if not image_path:
        return {"index": index, "group_id": group_id, "prediction": None}
    
    # Extract text from the image
    text = extract_text_from_image(image_path, enhance=False)
    if text is None or text.strip() == '':
        text = extract_text_from_image(image_path, enhance=True)
    
    # Extract the highest value unit from the text using entity name
    highest_unit = extract_highest_unit(entity_name, text)
    
    return {"index": index, "group_id": group_id, "prediction": highest_unit}

if __name__ == "__main__":
    # Load the dataset from the CSV
    csv_file_path = "sample_test.csv"
    df = pd.read_csv(csv_file_path, low_memory=False)

    # Encode group_id for clustering
    label_encoder = LabelEncoder()
    df['encoded_group_id'] = label_encoder.fit_transform(df['group_id'])

    # Perform KMeans clustering
    num_clusters = len(df['encoded_group_id'].unique())  # Number of clusters based on unique group_ids
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['encoded_group_id']])

    # Extract text and relevant units for each image using parallel processing
    results = []

    # ThreadPoolExecutor to process rows in parallel within each cluster
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predictor, row['image_link'], row['group_id'], row['entity_name'], index) 
                   for index, row in df.iterrows()]

        # Use tqdm to show progress
        for future in tqdm(futures, total=len(futures), desc="Extracting data", unit="image"):
            results.append(future.result())

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Extract the actual and predicted values
    y_true = df.set_index('group_id')['entity_value']
    y_pred = results_df.set_index('group_id')['prediction']

    # Ensure the indices match for comparison
    y_true = y_true.reindex(y_pred.index)
    y_pred = y_pred.reindex(y_true.index)

    # Drop any rows with NaN values in y_true or y_pred
    valid_indices = y_true.dropna().index.intersection(y_pred.dropna().index)
    y_true = y_true.loc[valid_indices]
    y_pred = y_pred.loc[valid_indices]

    # Convert values to strings if necessary (for categorical labels)
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    # Compute F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'binary', 'micro', 'macro', or 'weighted'

    print(f"F1 Score: {f1:.2f}")

    # Optionally, save results to CSV
    results_df.to_csv('results_with_f1_score.csv', index=False)
