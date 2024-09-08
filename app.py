from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
import pandas as pd
import os

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load TensorFlow model
try:
    model = tf.keras.models.load_model('C:/Users/Rishav/OneDrive/Desktop/Sih/Group-1/Tesd-2/FV.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Labels for classification
labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Route for home page
@app.route('/')
def home():
    return render_template("index.html")

# Function to get nutritional information
def get_nutrition(food_name):
    api_key = 'd4D6dSOc81pTAOY2gsNZ0YhjkMlhStLJRoII5SJu'
    nutrition_data = {}
    
    for name in food_name:
        url = f'https://api.nal.usda.gov/fdc/v1/foods/search?api_key={api_key}&query={name}'
        try:
            response = requests.get(url)
            data = response.json()
            flatten_json = pd.json_normalize(data["foods"])
            first_food = flatten_json.iloc[0]
            first_food_nutrition_list = first_food['foodNutrients']
            
            protein, calcium, fat, carbs, vitamin_a, vitamin_c = 0, 0, 0, 0, 0, 0
            
            for items in first_food_nutrition_list:
                if items['nutrientNumber'] == '203':  # Protein
                    protein = items['value']
                elif items['nutrientNumber'] == '301':  # Calcium
                    calcium = items['value']
                elif items['nutrientNumber'] == '204':  # Fat
                    fat = items['value']
                elif items['nutrientNumber'] == '205':  # Carbohydrates
                    carbs = items['value']
                elif items['nutrientNumber'] == '318':  # Vitamin A
                    vitamin_a = items['value']
                elif items['nutrientNumber'] == '401':  # Vitamin C
                    vitamin_c = items['value']
            
            vitamins = float(vitamin_a) + float(vitamin_c)
            
            nutrition_data[name] = {
                'protein': protein,
                'calcium': calcium / 1000,  # Converting calcium to grams
                'fat': fat,
                'carbs': carbs,
                'vitamins': vitamins / 1000  # Converting vitamins to grams
            }
        except Exception as e:
            print(f"Error fetching nutrition data for {name}: {e}")
            nutrition_data[name] = {'error': 'Failed to fetch data'}
    
    return nutrition_data

# Function to prepare image for prediction
def prepare_image(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions, axis=-1)[0]
        return labels[class_index]
    except Exception as e:
        print(f"Error in image processing: {e}")
        raise

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('results.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', error='No selected file')

    if file and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Predict
            result = prepare_image(file_path)
            print(f"Prediction result: {result}")

            # Fetch nutritional information
            nutrition_data = get_nutrition([result])
            print(f"Nutrition data: {nutrition_data.get(result, {})}")

            # Render result page
            return render_template('results.html', 
                                   prediction=result,
                                   nutrition=nutrition_data.get(result, {}),
                                   image_url=file_path)
        except Exception as e:
            print(f"Error processing file: {e}")
            return render_template('results.html', error='Error processing file')

# Route to handle manual input
@app.route('/manual', methods=['POST'])
def manual_input():
    food_name = request.form['food_name'].strip().lower()
    if not food_name:
        return render_template('results.html', error='No food name provided')

    try:
        # Fetch nutritional information
        nutrition_data = get_nutrition([food_name])
        if food_name in nutrition_data:
            return render_template('results.html', 
                                   prediction=food_name,
                                   nutrition=nutrition_data.get(food_name, {}))
        else:
            return render_template('results.html', 
                                   error='Nutritional data not found')
    except Exception as e:
        print(f"Error processing food name: {e}")
        return render_template('results.html', error='Error processing food name')

# Start the server
if __name__ == '__main__':
    # Check if the upload folder exists; if not, create it
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the Flask app on port 3000 with debug mode enabled
    app.run(port=3000, debug=True)
