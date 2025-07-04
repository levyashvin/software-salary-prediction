from flask import Flask, request, render_template
import pickle
import pandas as pd           # ← NEW

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Software Industry Salary Prediction.pkl', 'rb'))

# Column order used during training
FEATURE_ORDER = [
    'Rating',
    'Company Name',
    'Job Title',
    'Salaries Reported',
    'Location',
    'Employment Status',
    'Job Roles',
]

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # ------- 1️⃣  Collect form inputs -------
        data = {
            'Company Name'     : request.form['Company Name'],
            'Job Title'        : request.form['Job Title'],
            'Location'         : request.form['Location'],
            'Job Roles'        : request.form['Job Roles'],
            'Employment Status': request.form['Employment Status'],
            'Rating'           : float(request.form['Rating']),
            'Salaries Reported': int(request.form['Salaries Reported']),
        }

        # ------- 2️⃣  Build a DataFrame in training order -------
        input_df = pd.DataFrame([data])[FEATURE_ORDER]

        # ------- 3️⃣  Predict -------
        pred = model.predict(input_df)[0]
        pred = max(pred, 0)          # clip negatives if you like

        return render_template(
            'result.html',
            predict=f"The Predicted Salary of an Employer is: {pred:,.0f}"
        )

    return render_template('predict.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
