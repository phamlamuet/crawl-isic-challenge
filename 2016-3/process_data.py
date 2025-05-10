import os
import shutil
import pandas as pd

# Create destination directories
os.makedirs('melanoma_dataset/train/benign', exist_ok=True)
os.makedirs('melanoma_dataset/train/malignant', exist_ok=True)
os.makedirs('melanoma_dataset/test/benign', exist_ok=True)
os.makedirs('melanoma_dataset/test/malignant', exist_ok=True)


# Process training data
def organize_training_data():
    print("Organizing training data...")

    # Read training ground truth CSV
    train_gt_path = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
    train_df = pd.read_csv(train_gt_path, header=None)
    train_df.columns = ['image_id', 'label']

    # Source directory for training images
    train_images_dir = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3_Training_Data/ISBI2016_ISIC_Part3_Training_Data'

    # Copy files to appropriate folders
    for _, row in train_df.iterrows():
        image_id = row['image_id']
        label = row['label']

        # Find the image file (could be .jpg, .png, etc.)
        source_files = [f for f in os.listdir(train_images_dir)
                        if f.startswith(image_id)]

        if not source_files:
            print(f"Warning: Could not find image for {image_id}")
            continue

        source_file = os.path.join(train_images_dir, source_files[0])

        # Determine destination based on label
        if label == 'benign':
            dest_dir = 'melanoma_dataset/train/benign'
        else:  # 'malignant'
            dest_dir = 'melanoma_dataset/train/malignant'

        # Copy file
        dest_file = os.path.join(dest_dir, source_files[0])
        shutil.copy2(source_file, dest_file)

    # Count files in each category
    benign_count = len(os.listdir('melanoma_dataset/train/benign'))
    malignant_count = len(os.listdir('melanoma_dataset/train/malignant'))
    print(f"Training data organized: {benign_count} benign, {malignant_count} malignant")


# Process test data
def organize_test_data():
    print("Organizing test data...")

    # Read test ground truth CSV
    test_gt_path = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
    test_df = pd.read_csv(test_gt_path, header=None)
    test_df.columns = ['image_id', 'label']

    # Source directory for test images
    test_images_dir = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3_Test_Data/ISBI2016_ISIC_Part3_Test_Data'

    # Copy files to appropriate folders
    for _, row in test_df.iterrows():
        image_id = row['image_id']
        label = row['label']

        # Find the image file (could be .jpg, .png, etc.)
        source_files = [f for f in os.listdir(test_images_dir)
                        if f.startswith(image_id)]

        if not source_files:
            print(f"Warning: Could not find image for {image_id}")
            continue

        source_file = os.path.join(test_images_dir, source_files[0])

        # Determine destination based on label
        if label == 0.0:  # benign
            dest_dir = 'melanoma_dataset/test/benign'
        else:  # 1.0 = malignant
            dest_dir = 'melanoma_dataset/test/malignant'

        # Copy file
        dest_file = os.path.join(dest_dir, source_files[0])
        shutil.copy2(source_file, dest_file)

    # Count files in each category
    benign_count = len(os.listdir('melanoma_dataset/test/benign'))
    malignant_count = len(os.listdir('melanoma_dataset/test/malignant'))
    print(f"Test data organized: {benign_count} benign, {malignant_count} malignant")


# Run the organization functions
organize_training_data()
organize_test_data()

print("Dataset organization complete!")