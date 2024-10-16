import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper functions for K-Means
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

def save_cluster_images(image, labels, centroids, k, folder):
    cluster_filenames = []
    pixels = image.reshape(-1, 3)
    
    for i in range(k):
        cluster_image = np.zeros_like(pixels)
        cluster_image[labels == i] = pixels[labels == i]
        clustered_img = cluster_image.reshape(image.shape)

        filename = f'cluster_{i + 1}.png'
        filepath = os.path.join(folder, filename)
        Image.fromarray(clustered_img.astype(np.uint8)).save(filepath)
        cluster_filenames.append(filename)

    return cluster_filenames

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        k = int(request.form['k'])
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)

            # Load the image and process it
            image = np.array(Image.open(filepath))
            labels, centroids = kmeans(image, k)

            # Save each cluster as a separate image
            cluster_filenames = save_cluster_images(image, labels, centroids, k, app.config['UPLOAD_FOLDER'])

            # Render the result page
            return render_template('result.html', cluster_images=cluster_filenames)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
