import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import colorsys
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the Item model
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(10), nullable=False)  # "shirt" or "pants"
    category = db.Column(db.String(50), nullable=False)  # "general"
    style = db.Column(db.String(50), nullable=False)  # e.g., "t-shirt", "shirt", "printed shirt", "jeans", "formal trousers"
    image_path = db.Column(db.String(255), nullable=False)
    colors = db.Column(db.String(255))  # Comma-separated list of dominant colors (e.g., "blue,white")
    brightness = db.Column(db.String(10))  # "dark" or "pale"

    def to_dict(self, base_url):
        return {
            'id': self.id,
            'type': self.type,
            'category': self.category,
            'style': self.style,
            'image': f"{base_url}/uploads/{os.path.basename(self.image_path)}",
            'colors': self.colors.split(',') if self.colors else [],
            'brightness': self.brightness
        }

# Create the database
with app.app_context():
    db.create_all()

# Load the pre-trained TensorFlow model
try:
    model = load_model('clothing_classifier.h5')
    print("Pre-trained model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise Exception("Could not load the pre-trained model. Please ensure 'clothing_classifier.h5' exists.")

# Define class labels
CLASS_LABELS = ["t-shirt", "shirt", "printed shirt", "jeans", "formal trousers"]

# Function to preprocess image for TensorFlow model
def preprocess_image_for_model(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image for model: {e}")
        return None

# Function to detect item style using TensorFlow model
def detect_item_style(image_path, item_type):
    try:
        img_array = preprocess_image_for_model(image_path)
        if img_array is None:
            raise ValueError("Image preprocessing failed")

        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_style = CLASS_LABELS[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        print(f"Model prediction: {predicted_style} (confidence: {confidence:.2f})")

        if item_type == "shirt":
            if predicted_style not in ["t-shirt", "shirt", "printed shirt"]:
                filename = os.path.basename(image_path).lower()
                if 'tshirt' in filename or 'tee' in filename:
                    predicted_style = 't-shirt'
                elif 'printed' in filename or 'pattern' in filename:
                    predicted_style = 'printed shirt'
                else:
                    predicted_style = 'shirt'
        elif item_type == "pants":
            if predicted_style not in ["jeans", "formal trousers"]:
                filename = os.path.basename(image_path).lower()
                if 'jeans' in filename or 'denim' in filename:
                    predicted_style = 'jeans'
                else:
                    predicted_style = 'formal trousers'
        else:
            predicted_style = 'other'

        print(f"Detected style for {item_type} (image: {image_path}): {predicted_style}")
        return predicted_style
    except Exception as e:
        print(f"Error detecting style for {item_type} (image: {image_path}): {e}")
        filename = os.path.basename(image_path).lower()
        if item_type == 'shirt':
            if 'tshirt' in filename or 'tee' in filename:
                return 't-shirt'
            elif 'printed' in filename or 'pattern' in filename:
                return 'printed shirt'
            else:
                return 'shirt'
        elif item_type == 'pants':
            if 'jeans' in filename or 'denim' in filename:
                return 'jeans'
            else:
                return 'formal trousers'
        return 'other'

# Function to extract dominant colors and determine brightness using OpenCV
def extract_dominant_colors(image_path, num_colors=2):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        pixels = img.reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, palette = cv2.kmeans(
            pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        colors = palette.astype(int)

        color_names = []
        hsv_colors = []
        for color in colors:
            r, g, b = color
            r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
            h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
            hsv_colors.append((h * 360, s * 100, v * 100))

            if r > 200 and g < 100 and b < 100:
                color_names.append("red")
            elif r < 100 and g > 200 and b < 100:
                color_names.append("green")
            elif r < 100 and g < 100 and b > 200:
                color_names.append("blue")
            elif r > 200 and g > 200 and b < 100:
                color_names.append("yellow")
            elif r > 200 and g > 200 and b > 200:
                color_names.append("white")
            elif r < 50 and g < 50 and b < 50:
                color_names.append("black")
            else:
                color_names.append("other")

        dominant_color = colors[0]
        r, g, b = dominant_color
        brightness_value = 0.299 * r + 0.587 * g + 0.114 * b
        brightness = "dark" if brightness_value < 128 else "pale"

        return color_names, brightness, hsv_colors
    except Exception as e:
        print(f"Error extracting colors: {e}")
        return ["unknown"], "unknown", [(0, 0, 0)]

# Curated color pairing rules
COLOR_PAIRING_RULES = {
    ('blue', 'white'): 1.0,
    ('blue', 'black'): 0.9,
    ('blue', 'beige'): 0.8,
    ('black', 'white'): 1.0,
    ('black', 'gray'): 0.9,
    ('black', 'red'): 0.8,
    ('white', 'black'): 1.0,
    ('white', 'blue'): 1.0,
    ('white', 'gray'): 0.9,
    ('gray', 'black'): 0.9,
    ('gray', 'white'): 0.9,
    ('gray', 'blue'): 0.9,
    ('red', 'black'): 0.8,
    ('red', 'white'): 0.8,
    ('yellow', 'blue'): 0.8,
    ('yellow', 'black'): 0.7,
    ('green', 'white'): 0.6,
    ('green', 'black'): 0.7,
    ('beige', 'blue'): 0.8,
    ('beige', 'black'): 0.7,
    ('other', 'white'): 0.7,
    ('other', 'black'): 0.7,
    ('other', 'gray'): 0.7,
}

# Color matching algorithm
def get_color_compatibility(shirt_colors, shirt_hsv, pants_colors, pants_hsv):
    shirt_color = shirt_colors[0]
    pants_color = pants_colors[0]
    pair = (shirt_color, pants_color)
    reverse_pair = (pants_color, shirt_color)
    curated_score = COLOR_PAIRING_RULES.get(pair, COLOR_PAIRING_RULES.get(reverse_pair, 0.5))

    shirt_h, shirt_s, shirt_v = shirt_hsv[0]
    pants_h, pants_s, pants_v = pants_hsv[0]

    hue_diff = abs(shirt_h - pants_h)
    complementary_score = 0.7 if 150 <= hue_diff <= 210 else 0.0
    analogous_score = 1.0 if hue_diff <= 30 else 0.0
    monochromatic_score = 0.9 if hue_diff <= 10 and (abs(shirt_s - pants_s) > 20 or abs(shirt_v - pants_v) > 20) else 0.0

    is_shirt_neutral = shirt_s < 15 or shirt_v < 10 or shirt_v > 90 or shirt_color in ['white', 'black', 'gray']
    is_pants_neutral = pants_s < 15 or pants_v < 10 or pants_v > 90 or pants_color in ['white', 'black', 'gray']
    neutral_bonus = 0.8 if is_shirt_neutral or is_pants_neutral else 0.0

    hsv_score = max(complementary_score, analogous_score, monochromatic_score, 0.5) + neutral_bonus
    final_score = max(curated_score, hsv_score)
    final_score = min(final_score, 1.0)

    print(f"Color compatibility: Shirt Color={shirt_color} (HSV=({shirt_h}, {shirt_s}, {shirt_v})), Pants Color={pants_color} (HSV=({pants_h}, {pants_s}, {pants_v})), Curated Score={curated_score}, HSV Score={hsv_score}, Final Score={final_score}")
    return final_score

# Upload endpoint
@app.route('/api/upload', methods=['POST'])
def upload_item():
    print("Received upload request")
    if 'image' not in request.files or 'type' not in request.form:
        print("Missing image or type")
        return jsonify({'message': 'Image and type are required'}), 400

    file = request.files['image']
    type = request.form['type']
    category = request.form.get('category', 'general')

    print(f"Type: {type}, Category: {category}, File: {file.filename}")

    if type not in ['shirt', 'pants']:
        print("Invalid type")
        return jsonify({'message': 'Invalid type'}), 400

    filename = f"{type}_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Saving file to: {file_path}")
    file.save(file_path)

    style = detect_item_style(file_path, type)

    colors, brightness, hsv_colors = extract_dominant_colors(file_path)
    colors_str = ','.join(colors)
    print(f"Extracted colors: {colors_str}, Brightness: {brightness}, HSV: {hsv_colors}")

    new_item = Item(type=type, category=category, style=style, image_path=file_path, colors=colors_str, brightness=brightness)
    db.session.add(new_item)
    db.session.commit()
    print("Item saved to database")

    base_url = request.url_root.rstrip('/')
    return jsonify({
        'message': 'Item uploaded successfully',
        'item': new_item.to_dict(base_url)
    })

# Serve uploaded images
@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Get wardrobe endpoint
@app.route('/api/wardrobe', methods=['GET'])
def get_wardrobe():
    items = Item.query.all()
    base_url = request.url_root.rstrip('/')
    wardrobe = [item.to_dict(base_url) for item in items]
    return jsonify(wardrobe)

# Suggest outfit endpoint (modified to suggest 4 unique pairs with different shirts and pants)
@app.route('/api/suggest-outfit', methods=['GET'])
def suggest_outfit():
    occasion = request.args.get('occasion')
    if not occasion:
        return jsonify({'message': 'Occasion is required'}), 400

    all_shirts = Item.query.filter_by(type='shirt').all()
    all_pants = Item.query.filter_by(type='pants').all()

    print(f"Total shirts: {len(all_shirts)}, Total pants: {len(all_pants)}")
    for shirt in all_shirts:
        print(f"Shirt ID: {shirt.id}, Style: {shirt.style}, Brightness: {shirt.brightness}, Colors: {shirt.colors}")
    for pants in all_pants:
        print(f"Pants ID: {pants.id}, Style: {pants.style}, Brightness: {pants.brightness}, Colors: {pants.colors}")

    if not all_shirts or not all_pants:
        return jsonify({'message': 'Not available: No shirts or pants found.'}), 404

    # Define desired styles based on occasion
    if occasion == 'casual':
        desired_shirt_style = 't-shirt'
        desired_pants_style = 'jeans'
    elif occasion == 'formal':
        desired_shirt_style = 'shirt'
        desired_pants_style = 'formal trousers'
    elif occasion == 'party':
        desired_shirt_style = 'printed shirt'
        desired_pants_style = 'jeans'
    else:
        return jsonify({'message': 'Invalid occasion'}), 400

    # First try matching desired styles
    matching_shirts = [shirt for shirt in all_shirts if shirt.style == desired_shirt_style]
    matching_pants = [pants for pants in all_pants if pants.style == desired_pants_style]

    print(f"Matching shirts for {occasion} ({desired_shirt_style}): {len(matching_shirts)}")
    print(f"Matching pants for {occasion} ({desired_pants_style}): {len(matching_pants)}")

    # If no matching items, fall back to any shirts and pants
    if not matching_shirts:
        matching_shirts = all_shirts
        print(f"Fallback: Using all shirts since no {desired_shirt_style} found.")
    if not matching_pants:
        matching_pants = all_pants
        print(f"Fallback: Using all pants since no {desired_pants_style} found.")

    if not matching_shirts or not matching_pants:
        return jsonify({'message': f'Not available: No suitable shirts or pants found for {occasion}.'}), 404

    suggestions = []
    used_shirts = set()
    used_pants = set()
    base_url = request.url_root.rstrip('/')

    # Generate all possible pairs with color compatibility scores
    possible_pairs = []
    for shirt in matching_shirts:
        for pants in matching_pants:
            shirt_colors, _, shirt_hsv = extract_dominant_colors(shirt.image_path)
            pants_colors, _, pants_hsv = extract_dominant_colors(pants.image_path)
            score = get_color_compatibility(shirt_colors, shirt_hsv, pants_colors, pants_hsv)
            pair_info = (shirt, pants, score)
            possible_pairs.append(pair_info)
            print(f"Pair: Shirt (ID: {shirt.id}, Style: {shirt.style}, Brightness: {shirt.brightness}, Color: {shirt_colors[0]}), Pants (ID: {pants.id}, Style: {pants.style}, Brightness: {pants.brightness}, Color: {pants_colors[0]}), Color Score: {score}")

    print(f"Total possible pairs: {len(possible_pairs)}")

    # Sort pairs by color compatibility score (descending order)
    possible_pairs.sort(key=lambda x: x[2], reverse=True)

    # Select exactly 4 unique pairs (different shirts and different pants)
    for shirt, pants, score in possible_pairs:
        if len(suggestions) >= 4:  # Stop after selecting 4 pairs
            break
        if shirt.id in used_shirts or pants.id in used_pants:
            continue  # Skip if either shirt or pants is already used

        message = f'Color compatibility score: {score:.2f}'
        # Add a message if the style doesn't match the desired style
        if shirt.style != desired_shirt_style:
            message += f'. Using a non-preferred shirt style ({shirt.style}) for {occasion}.'
        if pants.style != desired_pants_style:
            message += f'. Using a non-preferred pants style ({pants.style}) for {occasion}.'
        # Add a message if brightness doesn't match the ideal
        if not ((shirt.brightness == 'dark' and pants.brightness == 'pale') or (shirt.brightness == 'pale' and pants.brightness == 'dark')):
            message += f'. Brightness combination ({shirt.brightness}-{pants.brightness}) may not be ideal.'

        suggestions.append({
            'shirt': shirt.to_dict(base_url),
            'pants': pants.to_dict(base_url),
            'message': message
        })
        used_shirts.add(shirt.id)
        used_pants.add(pants.id)
        print(f"Selected pair: Shirt ID {shirt.id} (Style: {shirt.style}, Color: {shirt.colors.split(',')[0]}), Pants ID {pants.id} (Style: {pants.style}, Color: {pants.colors.split(',')[0]}), Score: {score}, Message: {message}")

    if not suggestions:
        return jsonify({'message': f'Not available: No suitable pairs found for {occasion}.'}), 404

    return jsonify({
        'suggestions': suggestions,
        'message': f'Only {len(suggestions)} pairs available for {occasion}.' if len(suggestions) < 4 else None
    })

# Serve the front-end
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

# Serve static files (like back2.jpg) from the frontend directory
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)