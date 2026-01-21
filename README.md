# Automatic License Plate Detection and Recognition (ALPR)

This repository contains a course project developed for the Computer Vision course at **Technical University of Sofia – Branch Plovdiv**, within the C**omputer Systems and Technologies** program.

The project focuses on the implementation of an **Automatic License Plate Detection and Recognition (ALPR)** system using classical computer vision techniques and the **OpenCV** library.
The goal of the project is to apply fundamental image processing algorithms for license plate localization, character segmentation, and recognition, without the use of machine learning or deep learning methods.

---
The project uses CMake as its build system. First, create a build directory and run CMake to generate the build files. Then compile using CMake's build command. The system requires OpenCV and a C++17 compatible compiler.

For convenience, source the run.sh script which provides two functions: `release` for optimized builds and `debug` for development builds with diagnostic output. Both functions handle directory creation, configuration, and compilation automatically.

The compiled binary and KNN trainer tool will be placed in the build directory.

Simple steps:

# Build mdoes
when running
```bash
source run.sh
```

Output looks like this:
```bash
debug
  Builds with Debug mode
release
  Builds with Release mode
incremental-build
  Does incremental build with last build mode
  Can be built with a specific flags passed as arguments
```
It displays build modes and their descriptions.

# For release mode
Simply run:
```bash
release
```
Then in the same directory do:
```bash
./usr/bin/ALPR_System data/image1.jpg out/
```

# For debug mode
Simply run:
```bash
debug
```
Then in the same directory do:
```bash
./usr/bin/ALPR_System data/image1.jpg out/ --debug
```
