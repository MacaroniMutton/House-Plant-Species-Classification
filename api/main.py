# import os
# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved model
# MODEL_PATH = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\House_Plant_Species\\resnet_model.keras"
# model = load_model(MODEL_PATH)

# # Preprocessing function
# def preprocess_image(image, target_size=(224, 224)):
#     """
#     Resize and scale the input image.
#     """
#     image = image.resize(target_size)
#     image = np.array(image) / 255.0  # Rescale pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Define predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         # Open and preprocess the image
#         img = Image.open(file)
#         processed_image = preprocess_image(img)

#         # Make prediction
#         predictions = model.predict(processed_image)
#         predicted_class = np.argmax(predictions[0])
#         class_indices = {'African Violet (Saintpaulia ionantha)': 0, 'Aloe Vera': 1, 'Anthurium (Anthurium andraeanum)': 2, 'Areca Palm (Dypsis lutescens)': 3, 'Asparagus Fern (Asparagus setaceus)': 4, 'Begonia (Begonia spp.)': 5, 'Bird of Paradise (Strelitzia reginae)': 6, 'Birds Nest Fern (Asplenium nidus)': 7, 'Boston Fern (Nephrolepis exaltata)': 8, 'Calathea': 9, 'Cast Iron Plant (Aspidistra elatior)': 10, 'Chinese Money Plant (Pilea peperomioides)': 11, 'Chinese evergreen (Aglaonema)': 12, 'Christmas Cactus (Schlumbergera bridgesii)': 13, 'Chrysanthemum': 14, 'Ctenanthe': 15, 'Daffodils (Narcissus spp.)': 16, 'Dracaena': 17, 'Dumb Cane (Dieffenbachia spp.)': 18, 'Elephant Ear (Alocasia spp.)': 19, 'English Ivy (Hedera helix)': 20, 'Hyacinth (Hyacinthus orientalis)': 21, 'Iron Cross begonia (Begonia masoniana)': 22, 'Jade plant (Crassula ovata)': 23, 'Kalanchoe': 24, 'Lilium (Hemerocallis)': 25, 'Lily of the valley (Convallaria majalis)': 26, 'Money Tree (Pachira aquatica)': 27, 'Monstera Deliciosa (Monstera deliciosa)': 28, 'Orchid': 29, 'Parlor Palm (Chamaedorea elegans)': 30, 'Peace lily': 31, 'Poinsettia (Euphorbia pulcherrima)': 32, 'Polka Dot Plant (Hypoestes phyllostachya)': 33, 'Ponytail Palm (Beaucarnea recurvata)': 34, 'Pothos (Ivy arum)': 35, 'Prayer Plant (Maranta leuconeura)': 36, 'Rattlesnake Plant (Calathea lancifolia)': 37, 'Rubber Plant (Ficus elastica)': 38, 'Sago Palm (Cycas revoluta)': 39, 'Schefflera': 40, 'Snake plant (Sanseviera)': 41, 'Tradescantia': 42, 'Tulip': 43, 'Venus Flytrap': 44, 'Yucca': 45, 'ZZ Plant (Zamioculcas zamiifolia)': 46}
#         idx2label = list(class_indices.keys())
#         predicted_label = idx2label[predicted_class]

#         # Return prediction as JSON response
#         return jsonify({
#             "predicted_class": str(predicted_label),
#             "confidence": float(np.max(predictions[0]))
#         })

#     except Exception as e:
#         return jsonify({"error": "Error in prediction", "details": str(e)}), 500

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)

















import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and scale the input image.
    """
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Rescale pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# TensorFlow Serving URL
TF_SERVING_URL = "http://localhost:8501/v1/models/house_plant_model:predict"

# Define predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Open and preprocess the image
        img = Image.open(file)
        processed_image = preprocess_image(img)

        # Prepare the data payload for TensorFlow Serving
        data = {
            "instances": processed_image.tolist()
        }

        # Send a request to TensorFlow Serving
        response = requests.post(TF_SERVING_URL, json=data)
        
        if response.status_code != 200:
            return jsonify({"error": "Error in TensorFlow Serving", "details": response.text}), 500
        
        predictions = response.json()['predictions'][0]
        predicted_class = np.argmax(predictions)

        # Map predicted class to label
        class_indices = {'African Violet (Saintpaulia ionantha)': 0, 'Aloe Vera': 1, 'Anthurium (Anthurium andraeanum)': 2, 'Areca Palm (Dypsis lutescens)': 3, 'Asparagus Fern (Asparagus setaceus)': 4, 'Begonia (Begonia spp.)': 5, 'Bird of Paradise (Strelitzia reginae)': 6, 'Birds Nest Fern (Asplenium nidus)': 7, 'Boston Fern (Nephrolepis exaltata)': 8, 'Calathea': 9, 'Cast Iron Plant (Aspidistra elatior)': 10, 'Chinese Money Plant (Pilea peperomioides)': 11, 'Chinese evergreen (Aglaonema)': 12, 'Christmas Cactus (Schlumbergera bridgesii)': 13, 'Chrysanthemum': 14, 'Ctenanthe': 15, 'Daffodils (Narcissus spp.)': 16, 'Dracaena': 17, 'Dumb Cane (Dieffenbachia spp.)': 18, 'Elephant Ear (Alocasia spp.)': 19, 'English Ivy (Hedera helix)': 20, 'Hyacinth (Hyacinthus orientalis)': 21, 'Iron Cross begonia (Begonia masoniana)': 22, 'Jade plant (Crassula ovata)': 23, 'Kalanchoe': 24, 'Lilium (Hemerocallis)': 25, 'Lily of the valley (Convallaria majalis)': 26, 'Money Tree (Pachira aquatica)': 27, 'Monstera Deliciosa (Monstera deliciosa)': 28, 'Orchid': 29, 'Parlor Palm (Chamaedorea elegans)': 30, 'Peace lily': 31, 'Poinsettia (Euphorbia pulcherrima)': 32, 'Polka Dot Plant (Hypoestes phyllostachya)': 33, 'Ponytail Palm (Beaucarnea recurvata)': 34, 'Pothos (Ivy arum)': 35, 'Prayer Plant (Maranta leuconeura)': 36, 'Rattlesnake Plant (Calathea lancifolia)': 37, 'Rubber Plant (Ficus elastica)': 38, 'Sago Palm (Cycas revoluta)': 39, 'Schefflera': 40, 'Snake plant (Sanseviera)': 41, 'Tradescantia': 42, 'Tulip': 43, 'Venus Flytrap': 44, 'Yucca': 45, 'ZZ Plant (Zamioculcas zamiifolia)': 46}
        idx2label = list(class_indices.keys())
        predicted_label = idx2label[predicted_class]

        # Return prediction as JSON response
        return jsonify({
            "predicted_class": str(predicted_label),
            "confidence": float(np.max(predictions))
        })

    except Exception as e:
        return jsonify({"error": "Error in prediction", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
