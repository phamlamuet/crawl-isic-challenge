import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import glob

# Define the base directory and class names
base_dir = 'melanoma_dataset'
raw_dir = os.path.join(base_dir, 'raw')
organized_dir = os.path.join(base_dir, 'organized')

# Class names from the ground truth
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
splits = ['train', 'validation', 'test']

# Create organized directory structure
print("Creating directory structure...")
for split in splits:
    for class_name in class_names:
        os.makedirs(os.path.join(organized_dir, split, class_name), exist_ok=True)
print("Directory structure created!")


def find_csv_file(pattern):
    """Find CSV file using pattern matching"""
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None


def find_images_dir(pattern):
    """Find images directory using pattern matching - handle double nesting"""
    dirs = glob.glob(pattern, recursive=True)
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            # Check if this directory contains images directly
            image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            if image_files:
                return dir_path

            # Check for double-nested directories (most common case)
            nested_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            for nested_name in nested_dirs:
                nested_path = os.path.join(dir_path, nested_name)
                nested_images = glob.glob(os.path.join(nested_path, "*.jpg"))
                if nested_images:
                    print(f"Found double-nested images in: {nested_path}")
                    return nested_path
    return None


def organize_split(split_name, gt_file, images_dir):
    """Organize images for a specific split (train/validation/test)"""
    print(f"\nOrganizing {split_name} data...")

    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return 0, 0

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return 0, 0

    # Read ground truth CSV
    print(f"Reading ground truth file: {gt_file}")
    df = pd.read_csv(gt_file)

    # Print CSV info for debugging
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())

    # Check the actual values in the class columns
    print("\nChecking class column values...")
    for class_name in class_names:
        if class_name in df.columns:
            unique_vals = df[class_name].unique()
            print(f"{class_name}: {unique_vals}")

    # Determine the image column name
    image_col = None
    possible_image_cols = ['image', 'Image', 'ImageName', 'image_name', 'ID', 'id']
    for col in possible_image_cols:
        if col in df.columns:
            image_col = col
            break

    if image_col is None:
        # If no standard image column found, use the first column
        image_col = df.columns[0]
        print(f"Warning: No standard image column found, using first column: {image_col}")

    # Get list of available images
    print(f"Looking for images in: {images_dir}")
    available_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        available_images.extend(glob.glob(os.path.join(images_dir, ext)))

    # Create a mapping of base names to full paths
    image_mapping = {}
    for img_path in available_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image_mapping[base_name] = img_path

    print(f"Found {len(available_images)} images")
    print(f"Sample image files: {list(image_mapping.keys())[:5]}")

    # Process each row in the ground truth
    successful_moves = 0
    failed_moves = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        image_name = str(row[image_col]).strip()

        # Remove extension if present in the CSV
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_base_name = os.path.splitext(image_name)[0]
        else:
            image_base_name = image_name

        # Find which class this image belongs to
        image_class = None

        # Check for different possible values (1.0, 1, "1", True, etc.)
        for class_name in class_names:
            if class_name in df.columns:
                value = row[class_name]
                # Handle different formats: 1.0, 1, "1", True, etc.
                if value == 1.0 or value == 1 or value == "1" or value == "1.0" or value is True:
                    image_class = class_name
                    break

        if image_class is None:
            # Debug: print the row values to understand the format
            if idx < 5:  # Only print first 5 for debugging
                print(f"Debug row {idx}: {dict(row)}")
            failed_moves += 1
            continue

        # Look for the image file
        if image_base_name not in image_mapping:
            print(f"Warning: Image file not found for {image_name}")
            failed_moves += 1
            continue

        # Source and destination paths
        src_path = image_mapping[image_base_name]
        dst_path = os.path.join(organized_dir, split_name, image_class, os.path.basename(src_path))

        # Move the image
        try:
            shutil.copy2(src_path, dst_path)
            successful_moves += 1
        except Exception as e:
            print(f"Error moving {os.path.basename(src_path)}: {e}")
            failed_moves += 1

    print(f"{split_name} completed: {successful_moves} images moved, {failed_moves} failed")
    return successful_moves, failed_moves


# Define patterns to search for files (with recursive search)
splits_info = {
    'train': {
        'gt_pattern': os.path.join(raw_dir, '**', 'ISIC2018_Task3_Training_GroundTruth.csv'),
        'images_pattern': os.path.join(raw_dir, '*ISIC2018_Task3_Training_Input*'),
    },
    'validation': {
        'gt_pattern': os.path.join(raw_dir, '**', 'ISIC2018_Task3_Validation_GroundTruth.csv'),
        'images_pattern': os.path.join(raw_dir, '*ISIC2018_Task3_Validation_Input*'),
    },
    'test': {
        'gt_pattern': os.path.join(raw_dir, '**', 'ISIC2018_Task3_Test_GroundTruth.csv'),
        'images_pattern': os.path.join(raw_dir, '*ISIC2018_Task3_Test_Input*'),
    }
}

# Find actual file paths
print("Searching for ground truth files and image directories...")
actual_paths = {}
for split_name, patterns in splits_info.items():
    # Find ground truth file
    gt_file = find_csv_file(patterns['gt_pattern'])

    # Find images directory (handle double nesting)
    images_dir = find_images_dir(patterns['images_pattern'])

    if gt_file and os.path.exists(gt_file) and images_dir and os.path.exists(images_dir):
        actual_paths[split_name] = {
            'gt_file': gt_file,
            'images_dir': images_dir
        }
        print(f"{split_name}:")
        print(f"  Ground truth: {gt_file}")
        print(f"  Images: {images_dir}")
    else:
        print(f"Warning: Could not find files for {split_name}")
        if not gt_file or not os.path.exists(gt_file):
            print(f"  Missing ground truth file (pattern: {patterns['gt_pattern']})")
        if not images_dir or not os.path.exists(images_dir):
            print(f"  Missing images directory (pattern: {patterns['images_pattern']})")

# Organize each split
total_successful = 0
total_failed = 0

for split_name, paths in actual_paths.items():
    successful, failed = organize_split(split_name, paths['gt_file'], paths['images_dir'])
    total_successful += successful
    total_failed += failed

# Print summary statistics
print(f"\n{'=' * 50}")
print("ORGANIZATION SUMMARY")
print(f"{'=' * 50}")
print(f"Total images successfully organized: {total_successful}")
print(f"Total failed moves: {total_failed}")

# Print folder statistics
print(f"\n{'=' * 50}")
print("FOLDER STATISTICS")
print(f"{'=' * 50}")
for split in splits:
    print(f"\n{split.upper()}:")
    split_total = 0
    for class_name in class_names:
        class_dir = os.path.join(organized_dir, split, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {class_name}: {count} images")
            split_total += count
        else:
            print(f"  {class_name}: 0 images (directory not found)")
    print(f"  TOTAL {split}: {split_total} images")

print(f"\nDataset organized successfully!")
print(f"Organized data can be found in: {organized_dir}")