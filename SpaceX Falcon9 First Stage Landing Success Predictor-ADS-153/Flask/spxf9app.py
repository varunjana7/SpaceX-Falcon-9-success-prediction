import numpy as np
from flask import Flask, request, render_template
import pickle

# creating flask app
flask_app = Flask(__name__)
model = pickle.load(open("C:/Users/Pramo/SpaceX Falcon9 First Stage Landing Success Predictor-ADS-153/Flask/xgbmodel.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("homespxf9.html")

@flask_app.route("/predict")      
def predict():
    return render_template('predictspxf9.html')

@flask_app.route("/pred", methods=["POST"])
def pred():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = prediction[0]
    if output == 1:
        launch="Successful"
    else:
        launch="Unsuccessful"
    
    return render_template("finalspxf9.html", prediction_text = "Launch {}".format(launch))

if __name__ == "__main__":
    flask_app.run(debug=True)
