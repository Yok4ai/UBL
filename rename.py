#!/usr/bin/env python3
"""
Rename images in the Shelf directory to a clean, standardized format.
Handles duplicates, normalizes extensions, and removes special characters.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import re


def sanitize_filename(filename):
    """
    Clean up filename by removing special characters and normalizing.
    """
    # Remove special characters but keep alphanumeric, dots, underscores, hyphens
    name = re.sub(r'[^\w\s\-\.]', '_', filename)
    # Replace multiple spaces/underscores with single underscore
    name = re.sub(r'[\s_]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def normalize_extension(ext):
    """
    Normalize file extensions.
    """
    ext = ext.lower()
    # Convert jfif to jpg
    if ext == '.jfif':
        return '.jpg'
    # Convert jpeg to jpg
    if ext == '.jpeg':
        return '.jpg'
    return ext


def rename_images(directory, prefix=None, dry_run=True, keep_original=False):
    """
    Rename all images in directory to a clean format.

    Args:
        directory: Path to directory containing images
        prefix: Prefix for renamed files (default: None, just numbers)
        dry_run: If True, only show what would be renamed without actually renaming
        keep_original: If True, preserve original names in new filename
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    print("="*60)
    print("Image Renaming Tool")
    print("="*60)
    print(f"Directory: {directory}")
    print(f"Prefix: {prefix if prefix else 'None (just numbers)'}")
    print(f"Dry run: {dry_run}")
    print(f"Keep original names: {keep_original}")
    print("="*60)
    print()

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.gif'}
    image_files = []

    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)

    print(f"Found {len(image_files)} images")
    print()

    # Sort files by name for consistent ordering
    image_files.sort(key=lambda x: x.name.lower())

    # Track renamed files to handle duplicates
    renamed_count = defaultdict(int)
    renames = []

    for idx, file in enumerate(image_files, start=1):
        old_name = file.name
        old_stem = file.stem
        old_ext = file.suffix

        # Normalize extension
        new_ext = normalize_extension(old_ext)

        if keep_original:
            # Keep original name but sanitize it
            sanitized_stem = sanitize_filename(old_stem)
            if prefix:
                new_stem = f"{prefix}_{idx:04d}_{sanitized_stem}"
            else:
                new_stem = f"{idx:04d}_{sanitized_stem}"
        else:
            # Simple numbered naming
            if prefix:
                new_stem = f"{prefix}_{idx:04d}"
            else:
                new_stem = f"{idx:04d}"

        # Check for duplicates
        new_name = f"{new_stem}{new_ext}"
        if renamed_count[new_name] > 0:
            new_name = f"{new_stem}_{renamed_count[new_name]}{new_ext}"

        renamed_count[new_name] += 1

        new_path = directory / new_name

        # Store rename operation
        if old_name != new_name:
            renames.append((file, new_path, old_name, new_name))

    # Display renames
    if dry_run:
        print("DRY RUN - No files will be renamed")
        print("="*60)
        print()

    if renames:
        print(f"Renaming {len(renames)} files:")
        print()

        for old_path, new_path, old_name, new_name in renames:
            print(f"  {old_name}")
            print(f"    -> {new_name}")
            print()

            if not dry_run:
                try:
                    old_path.rename(new_path)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    print()

        if not dry_run:
            print("="*60)
            print(f"Successfully renamed {len(renames)} files!")
            print("="*60)
        else:
            print("="*60)
            print("To actually rename files, run with --execute flag")
            print("="*60)
    else:
        print("No files need renaming")

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rename images in Shelf directory to clean format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes) - just numbers
  python rename.py datasets/Shelf
  # Output: 0001.jpg, 0002.jpg, 0003.jpg, ...

  # Actually rename files (just numbers)
  python rename.py datasets/Shelf --execute

  # Use custom prefix
  python rename.py datasets/Shelf --prefix haircare --execute
  # Output: haircare_0001.jpg, haircare_0002.jpg, ...

  # Keep original names as part of new name
  python rename.py datasets/Shelf --keep-original --execute
  # Output: 0001_Hair_Care_1.jpg, 0002_Fabsol_Front_Side.jpg, ...

  # Prefix with original names
  python rename.py datasets/Shelf --prefix shelf --keep-original --execute
  # Output: shelf_0001_Hair_Care_1.jpg, shelf_0002_Fabsol_Front_Side.jpg, ...
        """
    )
    parser.add_argument('directory', type=str,
                       help='Path to directory containing images')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Prefix for renamed files (default: None, just numbers)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually rename files (default is dry run)')
    parser.add_argument('--keep-original', action='store_true',
                       help='Keep original names as part of new filename')
    return parser.parse_args()


def main():
    args = parse_args()

    rename_images(
        args.directory,
        prefix=args.prefix,
        dry_run=not args.execute,
        keep_original=args.keep_original
    )


if __name__ == '__main__':
    main()
