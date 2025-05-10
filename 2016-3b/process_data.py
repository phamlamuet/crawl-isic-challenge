import os
import shutil
import pandas as pd

# Create destination directories
os.makedirs('melanoma_dataset/train/benign', exist_ok=True)
os.makedirs('melanoma_dataset/train/malignant', exist_ok=True)
os.makedirs('melanoma_dataset/test/benign', exist_ok=True)
os.makedirs('melanoma_dataset/test/malignant', exist_ok=True)

# Create directories for segmentation files
os.makedirs('melanoma_dataset/train/benign_segmentation', exist_ok=True)
os.makedirs('melanoma_dataset/train/malignant_segmentation', exist_ok=True)
os.makedirs('melanoma_dataset/test/benign_segmentation', exist_ok=True)
os.makedirs('melanoma_dataset/test/malignant_segmentation', exist_ok=True)


# Process training data
def organize_training_data():
    print("Organizing training data...")

    # Read training ground truth CSV
    train_gt_path = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
    train_df = pd.read_csv(train_gt_path, header=None)
    train_df.columns = ['image_id', 'label']

    # Source directory for training images
    train_images_dir = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3B_Training_Data/ISBI2016_ISIC_Part3B_Training_Data'

    # Copy files to appropriate folders
    for _, row in train_df.iterrows():
        image_id = row['image_id']
        label = row['label']

        # Find the original image file
        image_files = [f for f in os.listdir(train_images_dir)
                       if f.startswith(image_id) and not '_Segmentation' in f]

        # Find the segmentation image file
        segmentation_files = [f for f in os.listdir(train_images_dir)
                              if f.startswith(image_id) and '_Segmentation' in f]

        if not image_files:
            print(f"Warning: Could not find original image for {image_id}")
            continue

        if not segmentation_files:
            print(f"Warning: Could not find segmentation image for {image_id}")
            continue

        # Get file paths
        image_file = os.path.join(train_images_dir, image_files[0])
        segmentation_file = os.path.join(train_images_dir, segmentation_files[0])

        # Determine destination based on label
        if label == 'benign':
            dest_dir = 'melanoma_dataset/train/benign'
            seg_dest_dir = 'melanoma_dataset/train/benign_segmentation'
        else:  # 'malignant'
            dest_dir = 'melanoma_dataset/train/malignant'
            seg_dest_dir = 'melanoma_dataset/train/malignant_segmentation'

        # Copy original image
        dest_file = os.path.join(dest_dir, image_files[0])
        shutil.copy2(image_file, dest_file)

        # Copy segmentation image
        seg_dest_file = os.path.join(seg_dest_dir, segmentation_files[0])
        shutil.copy2(segmentation_file, seg_dest_file)

    # Count files in each category
    benign_count = len(os.listdir('melanoma_dataset/train/benign'))
    malignant_count = len(os.listdir('melanoma_dataset/train/malignant'))
    benign_seg_count = len(os.listdir('melanoma_dataset/train/benign_segmentation'))
    malignant_seg_count = len(os.listdir('melanoma_dataset/train/malignant_segmentation'))

    print(f"Training data organized:")
    print(f"  - Benign: {benign_count} images, {benign_seg_count} segmentations")
    print(f"  - Malignant: {malignant_count} images, {malignant_seg_count} segmentations")


# Process test data
def organize_test_data():
    print("Organizing test data...")

    # Read test ground truth CSV
    test_gt_path = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv'
    test_df = pd.read_csv(test_gt_path, header=None)
    test_df.columns = ['image_id', 'label']

    # Source directory for test images
    test_images_dir = 'melanoma_dataset/raw/ISBI2016_ISIC_Part3B_Test_Data/ISBI2016_ISIC_Part3B_Test_Data'

    # Copy files to appropriate folders
    for _, row in test_df.iterrows():
        image_id = row['image_id']
        label = row['label']

        # Find the original image file
        image_files = [f for f in os.listdir(test_images_dir)
                       if f.startswith(image_id) and not '_Segmentation' in f]

        # Find the segmentation image file
        segmentation_files = [f for f in os.listdir(test_images_dir)
                              if f.startswith(image_id) and '_Segmentation' in f]

        if not image_files:
            print(f"Warning: Could not find original image for {image_id}")
            continue

        if not segmentation_files:
            print(f"Warning: Could not find segmentation image for {image_id}")
            continue

        # Get file paths
        image_file = os.path.join(test_images_dir, image_files[0])
        segmentation_file = os.path.join(test_images_dir, segmentation_files[0])

        # Determine destination based on label
        if label == 0.0:  # benign
            dest_dir = 'melanoma_dataset/test/benign'
            seg_dest_dir = 'melanoma_dataset/test/benign_segmentation'
        else:  # 1.0 = malignant
            dest_dir = 'melanoma_dataset/test/malignant'
            seg_dest_dir = 'melanoma_dataset/test/malignant_segmentation'

        # Copy original image
        dest_file = os.path.join(dest_dir, image_files[0])
        shutil.copy2(image_file, dest_file)

        # Copy segmentation image
        seg_dest_file = os.path.join(seg_dest_dir, segmentation_files[0])
        shutil.copy2(segmentation_file, seg_dest_file)

    # Count files in each category
    benign_count = len(os.listdir('melanoma_dataset/test/benign'))
    malignant_count = len(os.listdir('melanoma_dataset/test/malignant'))
    benign_seg_count = len(os.listdir('melanoma_dataset/test/benign_segmentation'))
    malignant_seg_count = len(os.listdir('melanoma_dataset/test/malignant_segmentation'))

    print(f"Test data organized:")
    print(f"  - Benign: {benign_count} images, {benign_seg_count} segmentations")
    print(f"  - Malignant: {malignant_count} images, {malignant_seg_count} segmentations")


# Run the organization functions
organize_training_data()
organize_test_data()

print("Dataset organization complete!")