from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras

app = Flask(__name__)

model = pickle.load(open('imgmodel.pickle',"rb"))

@app.route('/')
def home():
    return render_template("input2.html")

@app.route('/predict',methods=['POST'])
def predict():
    photos = UploadSet('photos', IMAGES)
    app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
    configure_uploads(app, photos)
    filename = photos.save(request.files['image'])
    image = keras.utils.load_img('./static/img'+filename, target_size=(224, 224))
    image = preprocess_input(image)
    prediction=model.predict(image)
    label = decode_predictions(prediction)
    result=label
    return render_template('input.html',resultat=f"Votre image est classifiée comme {result}")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    photos = UploadSet('photos', IMAGES)
    app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
    configure_uploads(app, photos)
    filename = photos.save(request.files['image'])
    image = keras.utils.load_img('./static/img'+filename, target_size=(224, 224))
    image = preprocess_input(image)
    prediction=model.predict(image)
    label = decode_predictions(prediction)
    result=label
    return jsonify(result)


if __name__ == '__main__':
    app.run()