import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = None

@app.route("/")
def home():
    global model
    if model is None:
        return redirect(url_for("train"))
    return render_template("index.html")


@app.route("/train")
def train():
    ufos = pd.read_csv("../data/ufos.csv")
    ufos = pd.DataFrame(
        {
            "Seconds": ufos["duration (seconds)"],
            "Country": ufos["country"],
            "Latitude": ufos["latitude"],
            "Longitude": ufos["longitude"],
        }
    )
    
    ufos.Country.unique()
    
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]
    
    ufos["Country"] = LabelEncoder().fit_transform(ufos["Country"])

    features = ["Seconds", "Latitude", "Longitude"]

    X = ufos[features]
    y = ufos["Country"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print("TRAINING")
    global model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return redirect("/")


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return redirect(url_for("train"))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    print(int_features)
    print(prediction)
    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
