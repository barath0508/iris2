from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session
model = joblib.load('iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            prediction = model.predict([features])[0]
            flower = label_encoder.inverse_transform([prediction])[0]
            session['result'] = f"Predicted Flower: {flower}"
        except:
            session['result'] = "Invalid input!"
        return redirect(url_for('output'))
    return render_template('index.html', result=None)

@app.route('/output')
def output():
    result = session.get('result', None)
    return render_template('output.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)