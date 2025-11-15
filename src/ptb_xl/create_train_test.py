'''#!/usr/bin/env python

# This file contains functions for creating training_data, vali_data, test_data:
#
#   python create_train_test.py \
#   -i input_data \
#   -d database_file \
#   -o output_folder
#
# where 'database_file' is the path to 'ptbxl_database.csv',
# 'input_data' is a folder containing the your data,
# 'outputs' is a folder for saving your outputs, and
#  -m is an optional argument to move files instead of copying them.

import json
import argparse
import glob
import os
import shutil
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.helper_code import *
from config import LEAD_LABEL_MAPPING


# Parse arguments.
def get_parser():
    description = "Run the data split."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--database_file", type=str, required=True)
    parser.add_argument("-i", "--input_data", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument(
        "-m", "--move", action="store_true", help="move files instead of copying"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        default=False,
        help="Whether to create masks",
    )
    parser.add_argument(
        "--mask_multilabel",
        action="store_true",
        default=False,
        help="Whether to multilabel classes",
    )
    parser.add_argument(
        "--rgba_to_rgb",
        action="store_true",
        default=False,
        help="Convert all rgba images to rgb images",
    )
    parser.add_argument(
        "--gray_to_rgb",
        action="store_true",
        default=False,
        help="Convert all gray scale images to rgb images",
    )
    parser.add_argument(
        "--rotate_image",
        action="store_true",
        default=False,
        help="Whether to rotate the images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument(
        "--plotted_pixels_key",
        type=str,
        default="plotted_pixels",
        help="Key for plotted pixels in json file",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        default=False,
        help="Whether to split or use all as training data",
    )
    return parser


# Function to either copy or move files
def transfer_file(file_path, target_dir, move):
    target_path = os.path.join(target_dir, os.path.basename(file_path))

    # Remove the file if it already exists
    if os.path.exists(target_path):
        os.remove(target_path)

    if move:
        shutil.move(file_path, target_dir)
    else:
        shutil.copy(file_path, target_dir)


# Run transfer in parallel
def parallel_transfer_files(source_paths, target_dir, move=False, num_workers=-1):
    if (num_workers == -1) or (num_workers > os.cpu_count()):
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    for source_path in source_paths:
        try:
            transfer_file(source_path, target_dir, move)
        except Exception as e:
            print(f"Error transferring file {source_path}: {e}")
            continue


# Function to convert rgba to rgb
def convert_images(file_path, rgba_to_rgb, rotate_image, original_folder_path):
    try:
        img = Image.open(file_path)
        if rgba_to_rgb and img.mode == "RGBA":
            img = img.convert("RGB")
        if rotate_image:
            img.save(original_folder_path)
            json_file = file_path.replace(".png", ".json")
            with open(json_file) as f:
                data_dict = json.load(f)
            rotation = data_dict["rotate"]
            img = img.rotate(rotation)
        img.save(file_path)
    except Exception as e:
        print(f"--------- ERROR IN {file_path} --------- {e} ---------")


# Run conversion in parallel
def convert_images_parallel(
    file_paths, rgba_to_rgb, rotate_image, original_folder_path, num_workers=-1
):
    if num_workers == -1:
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    for file_path, original_path in tqdm(
        zip(file_paths, original_folder_path), total=len(file_paths)
    ):
        try:
            convert_images(file_path, rgba_to_rgb, rotate_image, original_path)
        except Exception as e:
            print(f"Error converting image {file_path}: {e}")
            continue


# Function to create mask from json
def create_mask_from_json(
    json_path, mask_path, rgb=False, multilabel=False, plotted_pixels_key="plotted_pixels"
):
    try:
        # Get json info
        with open(json_path) as f:
            data_dict = json.load(f)

        # Check if augmented
        if "leads_augmented" in data_dict:
            mask_path_augmented = mask_path.replace(".png", "_augmented.png")
            keys_to_use = ["leads", "leads_augmented"]
            mask_paths_to_use = [mask_path, mask_path_augmented]
        else:
            keys_to_use = ["leads"]
            mask_paths_to_use = [mask_path]

        # Create mask
        mask_values = LEAD_LABEL_MAPPING
        full_mode_lead = data_dict["full_mode_lead"]
        for key, path_to_use in zip(keys_to_use, mask_paths_to_use):
            # Filter for full lead
            full_lead_length = max(
                [
                    lead["end_sample"] - lead["start_sample"]
                    for lead in data_dict[key]
                    if lead["lead_name"] == full_mode_lead
                ]
            )
            data_dict[key] = [
                lead
                for lead in data_dict[key]
                if lead["lead_name"] != full_mode_lead
                or lead["end_sample"] - lead["start_sample"] == full_lead_length
            ]

            # Get labels
            plotted_pixels = [
                (lead[plotted_pixels_key], lead["lead_name"]) for lead in data_dict[key]
            ]
            plotted_pixels = {
                tuple(np.array(item).astype("int")): subtuple[1]
                for subtuple in plotted_pixels
                for item in subtuple[0]
            }
            plotted_pixels = {
                k: v
                for k, v in plotted_pixels.items()
                if k[0] < data_dict["height"] and k[1] < data_dict["width"]
            }
            if multilabel:
                plotted_pixels = {k: mask_values[v] for k, v in plotted_pixels.items()}
            else:
                plotted_pixels = {k: 1 for k, v in plotted_pixels.items()}

            # Replace mask values with correct labels
            coords, values = zip(*plotted_pixels.items())
            coords = np.array(coords)
            values = np.array(values)
            rows, cols = coords[:, 0], coords[:, 1]
            mask = np.zeros((data_dict["height"], data_dict["width"]), dtype=np.uint8)
            mask[rows, cols] = values

            # Store
            if rgb:
                mask = np.stack([mask] * 3, axis=-1)
            Image.fromarray(mask).save(path_to_use)

    except Exception as e:
        print(f"--------- ERROR IN {json_path} --------- {e} ---------")


# Create masks in parallel
def create_mask_from_json_parallel(
    json_paths,
    mask_paths,
    rgb=False,
    multilabel=False,
    plotted_pixels_key="plotted_pixels",
    num_workers=-1,
):
    if num_workers == -1:
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    for json_path, mask_path in tqdm(zip(json_paths, mask_paths), total=len(json_paths)):
        try:
            create_mask_from_json(
                json_path, mask_path, rgb, multilabel, plotted_pixels_key
            )
        except Exception as e:
            print(f"Error creating mask for {json_path}: {e}")
            continue


# Run the code.
def run(args):
    # Get file paths
    if args.no_split:
        print(f"Only using training data for {args.input_data}...")
        data_groups = {"imagesTr": [], "imagesTv": [], "imagesTs": []}
        data_groups["imagesTr"] = glob.glob(f"{args.input_data}/**/*", recursive=True)
        count_files = len(data_groups["imagesTr"])
    else:
        print(f"Starting to determine data groups for {args.input_data}...")
        strat_fold_train = [1, 2, 3, 4, 5, 6, 7, 8]
        strat_fold_vali = [9]
        strat_fold_test = [10]
        dg = pd.read_csv(args.database_file, index_col="ecg_id")
        dg["file_start"] = dg.index.map(lambda x: str(x).zfill(5))
        data_groups = {"imagesTr": [], "imagesTs": [], "imagesTv": []}
        all_file_paths = glob.glob(f"{args.input_data}/**/*", recursive=True)
        count_files = 0
        for _, row in tqdm(dg.iterrows(), total=dg.shape[0]):
            file_start = row["file_start"]
            strat_fold = row["strat_fold"]
            matching_paths = [
                path for path in all_file_paths if f"{file_start}_lr" in path
            ]
            count_files += len(matching_paths)
            if strat_fold in strat_fold_train:
                data_groups["imagesTr"].extend(matching_paths)
            if strat_fold in strat_fold_vali:
                data_groups["imagesTv"].extend(matching_paths)
            if strat_fold in strat_fold_test:
                data_groups["imagesTs"].extend(matching_paths)
    print(
        f"In total splitted {len(data_groups['imagesTr'])} + {len(data_groups['imagesTv'])} + {len(data_groups['imagesTs'])} files, compared to {count_files} files in the input folder {args.input_data}."
    )

    # Create target directories and transfer images
    if args.move:
        print("Moving files...")
    else:
        print("Copying files...")
    for group_name, file_paths in tqdm(data_groups.items()):
        target_dir = os.path.join(args.output_folder, group_name)
        os.makedirs(target_dir, exist_ok=True)
        parallel_transfer_files(file_paths, target_dir, args.move, args.num_workers)

    # Optional: Convert all rgba to rgb and/or rotate images
    if args.rgba_to_rgb or args.rotate_image:
        if args.rgba_to_rgb and args.rotate_image:
            str_aux = "Converting images to rgb and rotating images"
        elif args.rgba_to_rgb:
            str_aux = "Converting images to rgb"
        else:
            str_aux = "Rotating images"
        for folder in ["imagesTr", "imagesTv", "imagesTs"]:
            print(f"{str_aux} for {folder}...")
            folder_path = os.path.join(args.output_folder, folder)
            possible_files = os.listdir(folder_path)
            file_options = [
                f"{f.split('/')[-1].split('_lr')[0]}_lr" for f in data_groups[folder]
            ]
            files_to_consider = [
                f
                for f in possible_files
                if any([f.startswith(fo) for fo in file_options])
            ]
            if args.rotate_image:
                original_folder_path = folder_path + "_original"
                os.makedirs(original_folder_path, exist_ok=True)
                original_file_paths = [
                    os.path.join(original_folder_path, file)
                    for file in files_to_consider
                    if file.endswith(".png")
                ]
                # Copy all json, hea and dat files to the original folder
                print("Copying original json, hea and dat files...")
                for file in tqdm(files_to_consider):
                    if (
                        file.endswith(".json")
                        or file.endswith(".hea")
                        or file.endswith(".dat")
                    ):
                        shutil.copy(
                            os.path.join(folder_path, file),
                            os.path.join(original_folder_path, file),
                        )
            else:
                original_file_paths = None
            file_paths_to_convert = [
                os.path.join(folder_path, file)
                for file in files_to_consider
                if file.endswith(".png")
            ]
            convert_images_parallel(
                file_paths_to_convert,
                args.rgba_to_rgb,
                args.rotate_image,
                original_file_paths,
                args.num_workers,
            )

    # Create masks
    if args.mask:
        for folder in ["imagesTr", "imagesTv", "imagesTs"]:
            print(f"Creating masks for {folder}...")
            file_options = [
                f"{f.split('/')[-1].split('_lr')[0]}_lr" for f in data_groups[folder]
            ]
            old_folder_path = os.path.join(args.output_folder, folder)
            new_folder_path = old_folder_path.replace("imagesT", "labelsT")
            os.makedirs(new_folder_path, exist_ok=True)
            json_files = [
                file
                for file in os.listdir(old_folder_path)
                if file.endswith(".json")
                and any([file.startswith(fo) for fo in file_options])
            ]
            mask_file_names = [
                os.path.join(new_folder_path, file.replace("_0000.json", ".png"))
                for file in json_files
            ]
            json_file_paths = [os.path.join(old_folder_path, file) for file in json_files]
            create_mask_from_json_parallel(
                json_file_paths,
                mask_file_names,
                args.gray_to_rgb,
                args.mask_multilabel,
                args.plotted_pixels_key,
                args.num_workers,
            )

    # Print number of files in each group
    print("Done with all processing, now counting files in each group...")
    num_training_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTr"))
            if file.endswith(".png")
        ]
    )
    num_validation_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTv"))
            if file.endswith(".png")
        ]
    )
    num_test_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTs"))
            if file.endswith(".png")
        ]
    )
    print(f"Training data: {num_training_data} images")
    print(f"Validation data: {num_validation_data} images")
    print(f"Test data: {num_test_data} images")

    # Create summary dataset.json file
    if args.mask_multilabel:
        labels_dict = {"background": 0}
        labels_dict.update(LEAD_LABEL_MAPPING)
    else:
        labels_dict = {"background": 0, "signal": 1}
    dataset_json_dict = {
        "channel_names": {"0": "Signals"},
        "labels": labels_dict,
        "numTraining": num_training_data,
        "file_ending": ".png",
    }
    with open(os.path.join(args.output_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json_dict, f)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
    print("Files have been transferred successfully.")
'''
# !/usr/bin/env python
"""
Simple mask creator - just give it a directory and it creates masks for all JSON files.

Usage:
    python simple_create_masks.py -i input_folder -o output_folder --plotted_pixels_key dense_plotted_pixels --mask_multilabel
"""

import json
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import LEAD_LABEL_MAPPING


def get_parser():
    parser = argparse.ArgumentParser(description="Create masks from JSON files")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Folder containing JSON files")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Folder to save masks")
    parser.add_argument("--plotted_pixels_key", type=str, default="plotted_pixels", help="Key for pixel data in JSON")
    parser.add_argument("--mask_multilabel", action="store_true", help="Use different labels for each lead")
    parser.add_argument("--gray_to_rgb", action="store_true", help="Convert grayscale to RGB")
    parser.add_argument("--dilate", type=int, default=0,
                        help="Dilate mask by N pixels to thicken lines (e.g., 1, 2, 3)")
    return parser


def create_mask_from_json(json_path, mask_path, rgb=False, multilabel=False, plotted_pixels_key="dense_plotted_pixels"):
    """Create a mask image from a JSON file."""
    try:
        # Load JSON
        with open(json_path) as f:
            data_dict = json.load(f)

        # Check for required keys
        if "leads" not in data_dict:
            print(f"WARNING: {json_path} missing 'leads' key. Keys: {list(data_dict.keys())}")
            return False

        # Check if augmented version exists
        if "leads_augmented" in data_dict:
            mask_path_augmented = mask_path.replace(".png", "_augmented.png")
            keys_to_use = ["leads", "leads_augmented"]
            mask_paths_to_use = [mask_path, mask_path_augmented]
        else:
            keys_to_use = ["leads"]
            mask_paths_to_use = [mask_path]

        # Get image dimensions
        height = data_dict.get("height", 1000)
        width = data_dict.get("width", 1000)

        # Process each lead set
        for key, path_to_use in zip(keys_to_use, mask_paths_to_use):
            if key not in data_dict:
                continue

            leads_data = data_dict[key]

            if not isinstance(leads_data, list) or len(leads_data) == 0:
                print(f"WARNING: {json_path} '{key}' is empty or not a list")
                continue

            # Collect all plotted pixels
            all_pixels = {}

            for lead in leads_data:
                if not isinstance(lead, dict):
                    continue

                # Get the pixel coordinates - check both keys
                if plotted_pixels_key in lead:
                    pixels = lead[plotted_pixels_key]
                elif "plotted_pixels" in lead:
                    print(f"  WARNING: Using 'plotted_pixels' fallback for {lead.get('lead_name', 'unknown')}")
                    pixels = lead["plotted_pixels"]
                else:
                    print(f"  WARNING: No pixel data found for {lead.get('lead_name', 'unknown')}")
                    continue

                lead_name = lead.get("lead_name", "unknown")

                print(f"  Lead {lead_name}: {len(pixels)} pixels from '{plotted_pixels_key}'")

                # Add each pixel to our dictionary
                for pixel in pixels:
                    if len(pixel) == 2:
                        row, col = int(pixel[0]), int(pixel[1])
                        # Check bounds
                        if 0 <= row < height and 0 <= col < width:
                            all_pixels[(row, col)] = lead_name

            if not all_pixels:
                print(f"WARNING: {json_path} no valid pixels found for '{key}'")
                continue

            print(f"Processing {json_path} ({key}): {len(all_pixels)} pixels")

            # Create mask array
            mask = np.zeros((height, width), dtype=np.uint8)

            # Fill in the mask
            if multilabel:
                # Different value for each lead
                for (row, col), lead_name in all_pixels.items():
                    mask[row, col] = LEAD_LABEL_MAPPING.get(lead_name, 1)
            else:
                # All signals get value 1
                for (row, col) in all_pixels.keys():
                    mask[row, col] = 1

            # Convert to RGB if requested
            if rgb:
                mask = np.stack([mask] * 3, axis=-1)

            # Save the mask
            #Image.fromarray(mask).save(path_to_use)

            # Create a visible version (scaled to 0-255)
            if mask.max() > 0:
                visible_mask = ((mask.astype(float) / mask.max()) * 255).astype(np.uint8)
                if rgb:
                    visible_mask = np.stack([visible_mask] * 3, axis=-1)
                visible_path = path_to_use.replace(".png", "_visible.png")
                Image.fromarray(visible_mask).save(visible_path)
                print(f"  → Saved {os.path.basename(path_to_use)} (max value: {mask.max()}, visible version saved)")

        return True

    except Exception as e:
        print(f"ERROR processing {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = get_parser().parse_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Find all JSON files
    json_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {args.input_folder}")
        return

    print(f"Found {len(json_files)} JSON files")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Pixel key: {args.plotted_pixels_key}")
    print(f"Multilabel: {args.mask_multilabel}")
    print("-" * 60)

    # Process each JSON file
    success_count = 0
    for json_file in tqdm(json_files, desc="Creating masks"):
        json_path = os.path.join(args.input_folder, json_file)
        mask_path = os.path.join(args.output_folder, json_file.replace('.json', '.png'))

        if create_mask_from_json(
                json_path,
                mask_path,
                rgb=args.gray_to_rgb,
                multilabel=args.mask_multilabel,
                plotted_pixels_key=args.plotted_pixels_key
        ):
            success_count += 1
    print(f"Successfully created masks for {success_count}/{len(json_files)} files")

    # Count output files
    mask_files = [f for f in os.listdir(args.output_folder) if f.endswith('.png')]
    print(f"Total mask files created: {len(mask_files)}")


if __name__ == "__main__":
    main()