import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Function to load the image and perform processing
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100))
    image_np = np.array(image)

    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    edges_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(edges_x, edges_y)

    return image_np, gray_image, edges_x, edges_y, magnitude

# Function to perform clustering
def cluster_image(image_path, k):
    image_np, gray_image, edges_x, edges_y, magnitude = load_image(image_path)

    pixels = image_np.reshape((-1, 3))

    # KMeans clustering using OpenCV
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    clustered_pixels = centers[labels.flatten()]
    clustered_image = clustered_pixels.reshape(image_np.shape)

    # Extracting the center pixel from the original and clustered images
    center_pixel_original = image_np[image_np.shape[0]//2, image_np.shape[1]//2]
    center_pixel_clustered = clustered_image[clustered_image.shape[0]//2, clustered_image.shape[1]//2]
    
    return image_np, clustered_image, edges_x, edges_y, magnitude, center_pixel_original, center_pixel_clustered

# Function to save gradient images
def save_gradient_images(edges_x, edges_y, magnitude):
    edge_x_path = os.path.join(app.config['OUTPUT_FOLDER'], 'grad_x.png')
    edge_y_path = os.path.join(app.config['OUTPUT_FOLDER'], 'grad_y.png')
    magnitude_path = os.path.join(app.config['OUTPUT_FOLDER'], 'magnitude.png')

    cv2.imwrite(edge_x_path, np.clip(edges_x, 0, 255).astype(np.uint8))
    cv2.imwrite(edge_y_path, np.clip(edges_y, 0, 255).astype(np.uint8))
    cv2.imwrite(magnitude_path, np.clip(magnitude, 0, 255).astype(np.uint8))

    return edge_x_path, edge_y_path, magnitude_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded.'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file.'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            k = int(request.form.get('k', 5))
            original_image, clustered_image, edges_x, edges_y, magnitude, center_pixel_original, center_pixel_clustered = cluster_image(file_path, k)

            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            Image.fromarray(clustered_image).save(output_path)

            edge_x_path, edge_y_path, magnitude_path = save_gradient_images(edges_x, edges_y, magnitude)
            return redirect(url_for('result', filename=filename, edge_x='grad_x.png', edge_y='grad_y.png', magnitude='magnitude.png',
                                    center_pixel_original=center_pixel_original.tolist(), center_pixel_clustered=center_pixel_clustered.tolist()))

    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    original_image_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    clustered_image_url = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    edge_x_url = os.path.join(app.config['OUTPUT_FOLDER'], 'grad_x.png')
    edge_y_url = os.path.join(app.config['OUTPUT_FOLDER'], 'grad_y.png')
    magnitude_url = os.path.join(app.config['OUTPUT_FOLDER'], 'magnitude.png')

    # Extracting the numpy values from the request parameters
    center_pixel_original = request.args.get('center_pixel_original')
    center_pixel_clustered = request.args.get('center_pixel_clustered')

    return render_template('result.html', 
                           original_image=original_image_url, 
                           clustered_image=clustered_image_url, 
                           edge_x=edge_x_url,
                           edge_y=edge_y_url,
                           magnitude=magnitude_url,
                           center_pixel_original=center_pixel_original,
                           center_pixel_clustered=center_pixel_clustered)

if __name__ == '__main__':
    app.run(debug=True)
