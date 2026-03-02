import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Models
MODEL_DIR = 'models'
cnn_extractor = None
pca_transformer = None
lr_model = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import joblib
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("WARNING: ML dependencies (tensorflow, joblib) not installed. Models cannot be loaded.")

def load_all_models():
    global cnn_extractor, pca_transformer, lr_model
    if not HAS_ML_DEPS:
        return
    try:
        print("Loading models...")
        cnn_path = os.path.join(MODEL_DIR, 'cnn_feature_extractor.h5')
        if not os.path.exists(cnn_path):
            print("Models not found! Training them now so predictions can work...")
            import train
            train.train_pipeline()

        cnn_extractor = load_model(cnn_path)
        pca_transformer = joblib.load(os.path.join(MODEL_DIR, 'pca_transformer.pkl'))
        lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Warning: Models not found or failed to load. Please run train.py first.\n{e}")

# Call load on startup
load_all_models()

def preprocess_image(base64_string):
    """ Converts base64 image from canvas to 28x28 normalized numpy array. """
    # Canvas contains transparent bg and black strokes. We need black bg and white strokes for MNIST.
    # The canvas data URL format is usually "data:image/png;base64,iVBOR..."
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGBA')
    
    # Create white background to blend transparent canvas into
    background = Image.new('RGBA', img.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, img)
    # Convert to grayscale
    gray_img = alpha_composite.convert('L')
    
    # Resize to 28x28
    img_resized = gray_img.resize((28, 28), resample=Image.Resampling.LANCZOS)
    
    # Invert colors (MNIST is white digits on black bg, we drew black on white)
    import PIL.ImageOps
    inverted_img = PIL.ImageOps.invert(img_resized)
    
    # Normalize pixel values
    img_array = np.array(inverted_img).astype('float32') / 255.0
    
    # Reshape for CNN (Batch, Height, Width, Channels)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.route('/')
def home():
    """ Serve the Frontend HTML. """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        if not (cnn_extractor and pca_transformer and lr_model):
            return jsonify({'error': 'Models are not loaded. Server needs to be trained first.'}), 503

        # 1. Preprocess the image
        img_array = preprocess_image(data['image'])
        
        # 2. Extract features with CNN
        cnn_features = cnn_extractor.predict(img_array, verbose=0)
        
        # 3. Reduce dimensions with PCA
        pca_features = pca_transformer.transform(cnn_features)
        
        # 4. Predict probabilities with Logistic Regression
        probabilities = lr_model.predict_proba(pca_features)[0]
        
        # Get top predicted class
        predicted_class = int(np.argmax(probabilities))
        
        # Format confidences
        confidences = []
        for digit, prob in enumerate(probabilities):
            confidences.append({
                'digit': digit,
                'conf': round(float(prob) * 100, 1) # Convert proportion to percentage map
            })
            
        # Sort by confidence
        confidences.sort(key=lambda x: x['conf'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidences': confidences
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
