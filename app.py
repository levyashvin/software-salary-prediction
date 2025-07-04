from flask import Flask, request, render_template
import pickle
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)

# ---------- Load Models ----------
model = pickle.load(open('models/Software Industry Salary Prediction.pkl', 'rb'))  # XGBoost (numeric)
band_model = CatBoostClassifier()
band_model.load_model('models/salary_band_model.cbm')  # CatBoost (salary band)

# ---------- Constants ----------
FEATURE_ORDER = [
    'Rating',
    'Company Name',
    'Job Title',
    'Salaries Reported',
    'Location',
    'Employment Status',
    'Job Roles',
]

CAT_COLS = [
    'Company Name',
    'Job Title',
    'Location',
    'Employment Status',
    'Job Roles',
]

# -----------------------------------------------
# Helper: Indian-style comma formatting + lakh-round
# -----------------------------------------------
def format_to_lakh(amount):
    """
    Round `amount` (rupees) to the nearest lakh (1 00 000) and
    return a string like 3,00,000.
    """
    rounded = round(amount / 100_000) * 100_000  # nearest lakh
    s = str(int(rounded))

    # Indian comma grouping: last 3 digits, then 2-digit groups
    if len(s) > 3:
        last3 = s[-3:]
        rest  = s[:-3][::-1]                    # reverse for easy slicing
        chunks = [rest[i:i+2] for i in range(0, len(rest), 2)]
        formatted = ",".join(chunks)[::-1] + "," + last3
    else:
        formatted = s
    return formatted

# Routes
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
        # Collect form inputs
        data = {
            'Company Name'     : request.form['Company Name'],
            'Job Title'        : request.form['Job Title'],
            'Location'         : request.form['Location'],
            'Job Roles'        : request.form['Job Roles'],
            'Employment Status': request.form['Employment Status'],
            'Rating'           : float(request.form['Rating']),
            'Salaries Reported': int(request.form['Salaries Reported']),
        }

        # Build DataFrame
        input_df = pd.DataFrame([data])[FEATURE_ORDER]

        # Numeric prediction
        pred_salary = model.predict(input_df)[0]
        pred_salary = max(pred_salary, 0)

        # round & format
        salary_formatted = format_to_lakh(pred_salary)

        # Band prediction (CatBoost)
        for col in CAT_COLS:
            input_df[col] = input_df[col].astype('category')

        pred_band = band_model.predict(input_df)[0]

        return render_template(
            'result.html',
            predict=f"Salary: ₹{salary_formatted} PA",
            band=f"Band: {pred_band[0]}"
        )

    return render_template('predict.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
