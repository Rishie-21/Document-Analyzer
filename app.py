from flask import Flask, render_template, request, make_response, send_file, flash
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance
from scipy.ndimage import gaussian_filter
import exifread
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import base64
import fitz
import tempfile
import PyPDF2
from datetime import datetime
import hashlib
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}

def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_pdf(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_PDF_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        file = request.files["file"]
        if file and allowed_image(file.filename):
            feature = request.form.get('feature', 'ela')
            if feature == 'ela':
                ela_im = perform_ela(file)
                buffer = BytesIO()
                ela_im.save(buffer, format="JPEG", quality=90)
                response = make_response(buffer.getvalue())
                response.headers["Content-Type"] = "image/jpeg"
            elif feature == 'histogram':
                buffer = histogram_analysis(Image.open(file))
                response = make_response(buffer.getvalue())
                response.headers["Content-Type"] = "image/png"
            elif feature == 'metadata':
                metadata = extract_metadata(file)
                return render_template("metadata.html", metadata=metadata)
            elif feature == 'noise':
                buffer, exaggerated_noise_image = noise_analysis_tool(Image.open(file))

                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                noise_buffer = BytesIO()
                exaggerated_noise_image.save(noise_buffer, format='PNG')
                noise_buffer.seek(0)
                noise_img_base64 = base64.b64encode(noise_buffer.getvalue()).decode('utf-8')

                return render_template('display_noise_analysis.html', histogram_data=img_base64, noise_image_data=noise_img_base64)
            elif feature == 'laplacian':
                variance = variance_of_laplacian(np.array(Image.open(file)))
                return f"Variance of Laplacian: {variance}"
            else:
                return "Invalid feature"
            return response
        elif file and allowed_pdf(file.filename):
            # Save the file temporarily
            file_descriptor, temp_file_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(file_descriptor, 'wb') as temp_file:
                temp_file.write(file.read())
                temp_file.seek(0) 

            # Extract font information
            font_info_list = extract_font_info(temp_file_path)

            # Analyze font information
            font_result = analyze_font_info(font_info_list)

            # Extract metadata
            with open(temp_file_path, 'rb') as temp_file:
                metadata = extract_pdf_metadata(temp_file)

                # Compute hashes
                metadata.update(compute_pdf_hashes(temp_file))

            # Remove the temporary file
            os.remove(temp_file_path)

            return render_template('pdf_analysis.html', font_result=font_result, metadata=metadata)

        else:
            return "Invalid file type. Please upload an image or a PDF file."

    return render_template("index.html")
# perform_ela, histogram_analysis, extract_metadata, noise_analysis_tool, variance_of_laplacian, extract_font_info, analyze_font_info
def perform_ela(image):
    im = Image.open(image)
    buffer = BytesIO()
    #save at quality 90% compression in the buffer
    im.save(buffer, format="JPEG", quality=90)
    #creatin ela_im in the buffer to create a 2ndimg object
    ela_im = Image.open(buffer)
    #calc diff bw compressed and uncompressed
    ela_im = ImageChops.difference(im, ela_im)
    #gettin extrema values of each pixel and raisin em to max value 255
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    #dividin 255 by max val and get the scalin factor and multiply with entire img also raise brightness&contrast
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im
def histogram_analysis(image):
    img = np.array(image)
    plt.figure()
    plt.hist(img.ravel(), 256, [0, 256])
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

def extract_metadata(image):
    img = Image.open(image)
    metadata = {
        "Dimensions": f"{img.width}x{img.height}",
        "File size": f"{len(image.read())} bytes",
        "File format": img.format
    }
    image.seek(0)
    tags = exifread.process_file(image)
    for tag, value in tags.items():
        if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
            metadata[tag] = str(value)
    return metadata

def local_standard_deviation(image, window_size=7):
    padded_image = np.pad(image, window_size // 2, mode='reflect')
    local_std = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_std[i, j] = np.std(padded_image[i:i + window_size, j:j + window_size])

    return local_std

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance
def noise_analysis_tool(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    local_std = local_standard_deviation(gray)
    exaggerated_noise = (local_std - np.min(local_std)) / (np.max(local_std) - np.min(local_std)) * 255
    exaggerated_noise = np.clip(exaggerated_noise, 0, 255).astype(np.uint8)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Grayscale Image')
    axs[1].hist(local_std.ravel(), bins=50, range=(0, np.percentile(local_std, 99)))
    axs[1].set_title('Histogram of Local Standard Deviations')
    axs[1].set_xlabel('Standard Deviation')
    axs[1].set_ylabel('Frequency')
    axs[1].set_yscale('log')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    exaggerated_noise_image = Image.fromarray(exaggerated_noise)
    return buffer, exaggerated_noise_image
def extract_font_info(pdf_file):
    doc = fitz.open(pdf_file)
    font_info_list = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_info_list.append({"font": span["font"], "size": span["size"], "color": span["color"],
                                               "text": span["text"], "page": page_num + 1})
    doc.close()

    return font_info_list
def analyze_font_info(font_info_list):
    if len(font_info_list) < 2:
        return {"result": "No variations detected."}

    result = {"result": "Variations detected:", "details": []}
    base_info = font_info_list[0]

    for info in font_info_list[1:]:
        changes = []
        if info["font"] != base_info["font"]:
            changes.append("font")
        if info["size"] != base_info["size"]:
            changes.append("font size")
        if info["color"] != base_info["color"]:
            changes.append("font color")

        if changes:
            result["details"].append({"page": info["page"], "text": info["text"], "changes": changes})

    return result
def extract_pdf_metadata(temp_file):
    temp_file.seek(0)
    pdf_reader = PyPDF2.PdfFileReader(temp_file)
    info = pdf_reader.getDocumentInfo()

    metadata = {
    'Title': info.get('/Title', None),
    'Author': info.get('/Author', None),
    'Creator': info.get('/Creator', None),
    'Producer': info.get('/Producer', None),
    'CreationDate': datetime.strptime(info['/CreationDate'][2:-7], '%Y%m%d%H%M%S') if '/CreationDate' in info else None,
    'ModDate': datetime.strptime(info['/ModDate'][2:-7], '%Y%m%d%H%M%S') if '/ModDate' in info else None
}
    return metadata


def compute_pdf_hashes(temp_file):
    temp_file.seek(0)
    file_content = temp_file.read()
    md5_hash = hashlib.md5(file_content).hexdigest()
    sha1_hash = hashlib.sha1(file_content).hexdigest()
    sha256_hash = hashlib.sha256(file_content).hexdigest()
 
    return {
        "MD5": md5_hash,
        "SHA-1": sha1_hash,
        "SHA-256": sha256_hash
    }

if __name__ == '__main__':
    app.run(debug=True)
