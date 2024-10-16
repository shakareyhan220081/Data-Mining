import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100))
    image_np = np.array(image)
    pixels = image_np.reshape((-1, 3))
    return pixels, image.size

def initialize_centroids(pixels, k):
    centroids = random.sample(list(pixels), k)
    return np.array(centroids)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(pixels, centroids):
    clusters = []
    for pixel in pixels:
        distances = [euclidean_distance(pixel, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def recalculate_centroids(pixels, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_pixels = pixels[clusters == i]
        if len(cluster_pixels) > 0:
            new_centroid = np.mean(cluster_pixels, axis=0)
        else:
            new_centroid = random.choice(pixels)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def kmeans(pixels, k, max_iters=100):
    centroids = initialize_centroids(pixels, k)
    for _ in range(max_iters):
        clusters = assign_clusters(pixels, centroids)
        new_centroids = recalculate_centroids(pixels, clusters, k)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def recreate_image(clusters, centroids, image_size):
    clustered_pixels = centroids[clusters].astype(np.uint8)
    clustered_image = clustered_pixels.reshape((image_size[1], image_size[0], 3))
    return clustered_image

def cluster_image(image_path, k):
    pixels, image_size = load_image(image_path)
    centroids, clusters = kmeans(pixels, k)
    clustered_image = recreate_image(clusters, centroids, image_size)
    return clustered_image

if __name__ == '__main__':
    image_path = 'image.jpg'
    k = 5
    clustered_image = cluster_image(image_path, k)
    
    original_image = Image.open(image_path)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    
    plt.subplot(1, 2, 2)
    plt.title(f'Clustered Image (k={k})')
    plt.imshow(clustered_image)
    plt.show()
