import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def initialize_centroids(image, k):
    pixels = image.reshape(-1, 3)
    random_indices = np.random.choice(pixels.shape[0], k, replace=False)
    return pixels[random_indices]

def assign_clusters(image, centroids):
    pixels = image.reshape(-1, 3)
    distances = np.sqrt(((pixels - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(image, labels, k):
    pixels = image.reshape(-1, 3)
    new_centroids = np.zeros((k, 3))
    for i in range(k):
        cluster_pixels = pixels[labels == i]
        new_centroids[i] = cluster_pixels.mean(axis=0) if len(cluster_pixels) > 0 else new_centroids[i]
    return new_centroids

def kmeans(image, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(image, k)
    for _ in range(max_iter):
        labels = assign_clusters(image, centroids)
        new_centroids = update_centroids(image, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return labels, centroids

def apply_segmentation(image, labels, centroids):
    segmented_image = centroids[labels].astype(np.uint8)
    return segmented_image.reshape(image.shape)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        max_clusters = int(request.form['max_clusters'])
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)

            # Load and process the uploaded image
            image = np.array(Image.open(filepath))
            segmented_images = []

            # Generate segmented images for each cluster from 1 to max_clusters
            for k in range(1, max_clusters + 1):
                labels, centroids = kmeans(image, k)
                segmented_image = apply_segmentation(image, labels, centroids)
                output_path = f'static/cluster_{k}.png'
                Image.fromarray(segmented_image).save(output_path)
                segmented_images.append(f'cluster_{k}.png')

            return render_template('index.html', segmented_images=segmented_images)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
