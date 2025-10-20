import os 
import pandas as pd
import PIL
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from skimage import io
import tensorflow as tf
import hashlib
from collections import defaultdict
import random
import shutil
import numpy as np
from tensorflow.keras import layers, models

def find_and_visualize_duplicates(folder_path, num_groups_to_show=3):
    """
    Finds exact duplicate images in a folder and visualizes the first few duplicate sets.
    
    Parameters:
        folder_path (str): Path to the folder containing images.
        num_groups_to_show (int): Number of duplicate groups to display.
    
    Returns:
        duplicates (dict): Dictionary mapping file hash to list of duplicate file paths.
    """
    # Build hash dictionary
    hash_dict = defaultdict(list)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                file_path = os.path.join(root, f)
                with open(file_path, "rb") as img_file:
                    file_hash = hashlib.md5(img_file.read()).hexdigest()
                hash_dict[file_hash].append(file_path)
    
    # Filter for duplicates
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    print(f"Found {len(duplicates)} sets of exact duplicates.")
    
    # Visualize some duplicate groups
    for h, paths in list(duplicates.items())[:num_groups_to_show]:
        plt.figure(figsize=(12, 4))
        for i, p in enumerate(paths):
            img = Image.open(p)
            plt.subplot(1, len(paths), i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"File {i+1}")
        plt.show()
    
    return duplicates

def create_filtered_dataset(all_image_paths, new_base_folder):
    """
    Create a new dataset folder from a list of image paths, preserving subfolder labels.

    Parameters:
        all_image_paths (list of str): Full paths to images to include.
        new_base_folder (str): Path to the new dataset folder.

    Returns:
        None
    """
    os.makedirs(new_base_folder, exist_ok=True)

    for img_path in all_image_paths:
        # Extract class name (parent folder name)
        class_name = os.path.basename(os.path.dirname(img_path))
        class_folder = os.path.join(new_base_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Destination path
        dest_path = os.path.join(class_folder, os.path.basename(img_path))
        
        # Copy image
        shutil.copy2(img_path, dest_path)  # copy2 preserves metadata

    print(f"Created new dataset at {new_base_folder} with {len(all_image_paths)} images.")

def find_duplicates_across_folders(folder1, folder2, num_groups_to_show=3):
    """
    Find exact duplicate images across two folders.

    Returns:
        duplicates (dict): {hash: [file_paths]} where each hash occurs more than once across folders.
    """
    hash_dict = defaultdict(list)
    
    for folder in [folder1, folder2]:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    file_path = os.path.join(root, f)
                    with open(file_path, "rb") as img_file:
                        file_hash = hashlib.md5(img_file.read()).hexdigest()
                    hash_dict[file_hash].append(file_path)
    
    # Only keep hashes with more than one file (duplicates)
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    print(f"Found {len(duplicates)} sets of duplicates across folders.")

    # Visualize duplicates
    for h, paths in list(duplicates.items())[:num_groups_to_show]:
        plt.figure(figsize=(12, 4))
        for i, p in enumerate(paths):
            img = Image.open(p)
            plt.subplot(1, len(paths), i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(p))
        plt.show()
    
    return duplicates


def get_class_distribution(folder_path):
    """
    Count the number of images per class (subfolder) in a dataset folder.
    
    Parameters:
        folder_path (str): Path to dataset folder with subfolders as class labels.
        
    Returns:
        class_counts (dict): Dictionary {class_name: num_images}.
    """
    class_counts = Counter()
    
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            num_images = len([
                f for f in os.listdir(class_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
            ])
            class_counts[class_name] = num_images
    
    return dict(class_counts)


def show_random_samples_per_class(dataset_folder, samples_per_class=10):
    """
    Display a random sample of images from each class in the dataset folder.

    Parameters:
        dataset_folder (str): Path to dataset folder with subfolders as classes.
        samples_per_class (int): Number of images to display per class.
    """
    classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    
    for class_name in classes:
        class_folder = os.path.join(dataset_folder, class_name)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        
        # Randomly sample images
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        plt.figure(figsize=(12, 3))
        plt.suptitle(f"Class: {class_name}", fontsize=16)
        
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path)
            
            plt.subplot(1, len(sample_images), i+1)
            plt.imshow(img,cmap="gray")
            plt.axis("off")
            plt.title(f"{i+1}")
        
        plt.show()


def resize_and_convert_to_grayscale(source_folder, dest_folder, target_size=(224, 224)):
    """
    Resize and convert all images in source_folder to grayscale, saving to dest_folder.

    Parameters:
        source_folder (str): Path to the dataset folder (with subfolders as classes)
        dest_folder (str): Path where converted images will be saved
        target_size (tuple): Desired image size (width, height)
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    # Iterate over class subfolders
    for class_name in os.listdir(source_folder):
        class_src_folder = os.path.join(source_folder, class_name)
        if not os.path.isdir(class_src_folder):
            continue
        
        class_dest_folder = os.path.join(dest_folder, class_name)
        os.makedirs(class_dest_folder, exist_ok=True)
        
        for img_name in os.listdir(class_src_folder):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                img_path = os.path.join(class_src_folder, img_name)
                img = Image.open(img_path)
                
                # Convert to grayscale
                img = img.convert("L")  # "L" mode is grayscale
                
                # Resize
                img = img.resize(target_size)
                
                # Save to new folder
                img.save(os.path.join(class_dest_folder, img_name))

    print(f"Processed images saved to {dest_folder}")


def plot_pixel_distribution(dataset_folder, sample_limit=None):
    """
    Plot pixel intensity distributions across images in a dataset folder.

    Parameters:
        dataset_folder (str): Path to dataset folder with subfolders as classes
        sample_limit (int): Max number of images to sample for plotting (for speed)
    """
    pixel_values = []

    # Iterate over class subfolders
    for class_name in os.listdir(dataset_folder):
        class_folder = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        
        # Limit number of images if requested
        if sample_limit:
            images = images[:sample_limit]

        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).convert("L")  # grayscale
            pixels = np.array(img).flatten()
            pixel_values.extend(pixels)

    pixel_values = np.array(pixel_values)

    plt.figure(figsize=(10, 6))
    plt.hist(pixel_values, bins=256, range=(0, 255), color='gray', alpha=0.7)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.title("Pixel intensity distribution across training dataset")
    plt.show()

def plot_pixel_distribution_no_background(dataset_folder, sample_limit=None):
    """
    Plot pixel intensity distributions across images in a dataset folder,
    ignoring black background pixels (value=0).

    Parameters:
        dataset_folder (str): Path to dataset folder with subfolders as classes
        sample_limit (int): Max number of images to sample for plotting (for speed)
    """
    pixel_values = []

    # Iterate over class subfolders
    for class_name in os.listdir(dataset_folder):
        class_folder = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        
        # Limit number of images if requested
        if sample_limit:
            images = images[:sample_limit]

        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).convert("L")  # grayscale
            pixels = np.array(img).flatten()
            
            # Ignore black background pixels
            pixels = pixels[pixels > 0]
            
            pixel_values.extend(pixels)

    pixel_values = np.array(pixel_values)

    plt.figure(figsize=(10, 6))
    plt.hist(pixel_values, bins=256, range=(1, 255), color='gray', alpha=0.7)  # start at 1
    plt.xlabel("Pixel intensity (excluding black background)")
    plt.ylabel("Frequency")
    plt.title("Pixel intensity distribution across dataset (background excluded)")
    plt.show()


def get_pixel_stats_ignore_background(dataset_folder, sample_limit=None):
    all_pixels = []

    for class_name in os.listdir(dataset_folder):
        class_folder = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        if sample_limit:
            images = images[:sample_limit]

        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).convert("L")
            pixels = np.array(img).flatten()

            # Exclude background pixels (0)
            pixels = pixels[pixels > 0]
            all_pixels.extend(pixels)

    all_pixels = np.array(all_pixels)
    stats = {
        'mean': np.mean(all_pixels),
        'std': np.std(all_pixels),
        'p5': np.percentile(all_pixels, 5),
        'p95': np.percentile(all_pixels, 95)
    }
    return stats


def flag_outlier_images(dataset_folder, normal_min, normal_max):
    """
    Returns list of image paths whose pixel values fall outside the normal range.
    """
    outlier_images = []

    for class_name in os.listdir(dataset_folder):
        class_folder = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).convert("L")
            pixels = np.array(img).flatten()
            if pixels.min() < normal_min or pixels.max() > normal_max:
                outlier_images.append(img_path)

    return outlier_images


def prepare(resize_and_rescale, data_augmentation, AUTOTUNE, ds, shuffle=False, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)


  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)



