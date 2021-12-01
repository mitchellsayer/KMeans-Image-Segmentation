# KMeans-Image-Segmentation
Implementation of K-Means Clustering for Image Segmentation

This package will provide an implementation of K-Means clustering applied to the problem of Image segmentation.

## Planned features of this package:
- Implementations of various distance metrics (not just euclidian)
- Automatic detection of optimal "K" value using elbow method
- Use OpenCV for fast processing of image data
- This package will be able to be imported and used in code (Publish on PyPi?)
- Easy to use CLI interface for segmentation of arbitrary images & simple integration into larger systems.

## Installation:
- `pip install -r requirements.txt`
## Running:
To run this project:
1. Modify config.json to specify input file path, output save path, and kmeans params.
2. Run the following command: `python3 segment_image.py`

