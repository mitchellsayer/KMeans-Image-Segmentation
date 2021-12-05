# KMeans-Image-Segmentation
Implementation of K-Means Clustering for Image Segmentation

This package implements K-Means clustering in Python3 & also contains a module which utilizes the k-means implementation to perform image segmentation. 

To run, place input files in images/inputs and update config.json with desired image path and other k means parameters.

## Installation:
- `pip install -r requirements.txt`

## Running:
To run this project:
1. Modify config.json to specify:
  - K (number of clusters)
  - Tolerance (converges once all cluster means move less than this amount in one iteration)
  - Maximum iterations
  - Random seed
  - Input file path
  - Output save path
3. Run the following command: `python3 segment_image.py`
