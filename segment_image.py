import sys
import json

import numpy as np
from PIL import Image

from src import KMeans

''' 
This will be the executable file which standalone users can run 
to segment individual images (or batches of images?)

The CLI interface is as follows:

python3 segment_image.py K INPUT_PATH OUTPUT_PATH
    - K (int): Number of clusters
    - INPUT_PATH: relative path to find input .jpg file
    - OUTPUT_PATH: relative path to save output .jpg file
'''

'''
Function to convert PIL Image to a normalized np.ndarray 
of the following shape:
    (image_width * image_height, 5)
Where the second dimension represents: (x, y, r, g, b)
'''

def parseConfig(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    return cfg

def get_image_data(image):
    image_width = image.size[0]
    image_height = image.size[1]
    image_data = np.ndarray(shape=(image_width * image_height, 5),
                            dtype=float)

    for x in range(0, image_width):
        for y in range(0, image_height):
            xy = (x, y)
            rgb = image.getpixel(xy)
            data_idx = x + (y * image_width)

            image_data[data_idx, 0] = x
            image_data[data_idx, 1] = y
            image_data[data_idx, 2] = rgb[0]
            image_data[data_idx, 3] = rgb[1]
            image_data[data_idx, 4] = rgb[2]

    return image_data

def construct_segmented_image(cluster_allocations, computed_centroids, input_image):
    image_width = input_image.size[0]
    image_height = input_image.size[1]
    processed_image_data = np.ndarray(shape=(image_width * image_height, 5),
                            dtype=float)
    
    # 
    for i, cluster in enumerate(cluster_allocations):
        centroid = computed_centroids[cluster]
        processed_image_data[i][2] = int(round(centroid[2] * 255))
        processed_image_data[i][3] = int(round(centroid[3] * 255))
        processed_image_data[i][4] = int(round(centroid[4] * 255))

    # Save image
    output_image = Image.new("RGB", (image_width, image_height))

    for x in range(image_width):
        for y in range(image_height):
            output_image.putpixel(
                (x, y),
                (
                    int(processed_image_data[y * image_width + x][2]),
                    int(processed_image_data[y * image_width + x][3]),
                    int(processed_image_data[y * image_width + x][4]),
                ),
            )
    
    return output_image

def main():
    # Parse Config
    cfg = parseConfig('./config.json')

    K = int(cfg['K'])
    if K < 2:
        print('K must be >=2')
        sys.exit()

    tolerance = float(cfg['TOLERANCE'])
    max_iterations = int(cfg['MAX_ITERATIONS'])
    random_seed = int(cfg['RANDOM_SEED'])

    input_path = cfg['INPUT_PATH']
    output_path = cfg['OUTPUT_PATH']
    
    # Process Input Image
    input_image = Image.open(input_path)
    image_data = get_image_data(input_image)
    input_image.show()

    kmeans_model = KMeans(n_clusters=K, 
                          tolerance=tolerance,
                          max_iterations=max_iterations,
                          random_seed=random_seed)
    
    clusters = kmeans_model.fit(image_data)

    output_image = construct_segmented_image(kmeans_model.cluster_allocations, 
                                             kmeans_model.centroids, 
                                             input_image)
    output_image.save(output_path)
    output_image.show()

if __name__ == "__main__":
    main()
