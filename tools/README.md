# KNN Training Tool

A utility for generating KNN training data (`classifications.xml` and `images.xml`) from a dataset of character images.

## Why Use This Tool?

Instead of manually labeling individual characters by typing them one-by-one, this tool automatically processes your existing dataset folder where images are organized by character class.

## Dataset Structure

Your training dataset should be organized like this:

```
dataset/
├── 0/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
├── 1/
├── 2/
├── ...
├── 9/
├── A/
│   ├── char_001.jpg
│   ├── char_002.jpg
│   └── ...
├── B/
├── ...
└── Z/
```

**Key points:**
- Each subfolder name is the character label (0-9, A-Z)
- Folder names are case-insensitive (both `a/` and `A/` work)
- Images can be any common format: PNG, JPG, JPEG, BMP, TIFF
- Images are automatically converted to grayscale and resized to 20×30 pixels

## Building

```bash
cd tools
mkdir -p build && cd build
cmake ..
make
```

## Usage

```bash
./KNN_Trainer <dataset_path> [output_dir]
```

**Arguments:**
- `dataset_path` - Path to your training dataset folder (required)
- `output_dir` - Where to save the XML files (default: current directory)

**Examples:**

```bash
# Generate XML files in current directory
./KNN_Trainer /path/to/my/dataset

# Generate XML files in the main project directory
./KNN_Trainer /path/to/my/dataset ../

# Generate XML files in a specific output folder
./KNN_Trainer /path/to/my/dataset /path/to/output
```

## Output Files

The tool generates two XML files:

1. **`classifications.xml`** - Contains ASCII codes for each character label
2. **`images.xml`** - Contains flattened image data (600 float values per image, representing a 20×30 pixel grayscale image)

## Using the Generated Files

Copy both XML files to your main ALPR project directory:

```bash
cp classifications.xml images.xml ../
```

Then rebuild and run the ALPR system as usual.

## How It Works

1. **Scans** all subfolders in the dataset directory
2. **Validates** folder names as valid character labels (0-9, A-Z)
3. **Processes** each image:
   - Reads in grayscale
   - Resizes to 20×30 pixels
   - Applies Otsu thresholding for binarization
   - Flattens to a 600-element vector
4. **Saves** training data to OpenCV XML format
5. **Verifies** by training a KNN classifier and running self-tests

## Tips for Good Training Data

- **Quantity**: More samples per character = better accuracy (aim for 50-100+ per character)
- **Variety**: Include different fonts, sizes, and slight rotations
- **Quality**: Clean, well-cropped images of individual characters
- **Balance**: Try to have similar sample counts for each character
- **Preprocessing**: Images should have dark characters on light backgrounds (or they will be automatically inverted)

## Example Output

```
========================================
   KNN Training Tool for ALPR System
========================================

Processing dataset from: /path/to/dataset

  [0] Processed 120 samples
  [1] Processed 115 samples
  ...
  [A] Processed 98 samples
  [B] Processed 102 samples
  ...

----------------------------------------
Processing complete!
  Total samples: 3240
  Skipped: 0
  Unique characters: 36
----------------------------------------

Saving training data...
  Saved: ./classifications.xml
  Saved: ./images.xml

Verifying training data...
  KNN training verification: SUCCESS

Quick self-test (testing on training data):
  Sample 142: 'A' -> 'A' ✓
  Sample 891: '7' -> '7' ✓
  ...

  Self-test accuracy: 10/10 (100%)

========================================
Training complete!
========================================
```
