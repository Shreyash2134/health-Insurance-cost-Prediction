from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# ==============================
# LOAD MODEL
# ==============================
with open("rf_tuned.pkl", 'rb') as file:
    model = pickle.load(file)

# ==============================
# HELPER FUNCTIONS
# ==============================
def encode_gender(g):
    return 1 if g.lower() in ['male', 'm'] else 0

def encode_smoker(s):
    return 1 if s.lower() == 'yes' else 0

def encode_region(r):
    region_map = {
        'northeast': 0,
        'northwest': 1,
        'southeast': 2,
        'southwest': 3
    }
    return region_map.get(r.lower(), 1)

def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)

# ==============================
# LIC LOGIC
# ==============================
def adjust_risk(cost, data):

    if data.get('alcoholic','no') == 'yes':
        cost *= 1.1

    disease = data.get('disease','none').lower()

    if disease == 'diabetes':
        cost *= 1.3
    elif disease == 'heart':
        cost *= 1.6

    if float(data.get('duration',0)) > 2:
        cost *= 1.1

    if data.get('treatment','none') == 'ongoing':
        cost *= 1.2

    return cost

def calculate_premium(cost):
    return round((cost / 100000) * 6000, 2)

def underwriting(cost):
    if cost > 600000:
        return "Rejected"
    elif cost > 300000:
        return "High Premium"
    return "Approved"

def calculate_claim(cost):
    return round(min(cost, 500000), 2)

# ==============================
# CLEAN CSV DATA
# ==============================
def clean_data(df):

    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
        'smoke': 'smoker',
        'drink': 'alcoholic',
        'duration of disease': 'duration',
        'duration_disease': 'duration'
    }, inplace=True)

    df['disease'] = df['disease'].fillna('none').str.lower()
    df['disease'] = df['disease'].replace({'no': 'none'})

    df['treatment'] = df['treatment'].astype(str).str.lower()
    df['treatment'] = df['treatment'].replace({
        'no': 'none',
        'tablet': 'ongoing',
        'insulin': 'ongoing'
    })

    def convert_duration(x):
        x = str(x).lower()
        if 'year' in x:
            return float(x.split()[0].replace('+',''))
        elif 'month' in x:
            return float(x.split()[0]) / 12
        return 0

    df['duration'] = df['duration'].apply(convert_duration)

    return df

# ==============================
# ROUTES
# ==============================
@app.route('/')
def home():
    return render_template('home.html')


# -------- FORM PREDICTION --------
@app.route('/predict', methods=['POST'])
def predict():

    data = request.form

    age = int(data['age'])
    height = float(data['height'])
    weight = float(data['weight'])
    children = int(data['children'])

    bmi = calculate_bmi(height, weight)

    sex = encode_gender(data['gender'])
    smoker = encode_smoker(data['smoker'])
    region = encode_region(data['region'])

    final = np.array([[age, sex, bmi, children, smoker, region]])

    cost = model.predict(final)[0]

    adj_cost = adjust_risk(cost, data)

    premium = calculate_premium(adj_cost)
    decision = underwriting(adj_cost)
    claim = calculate_claim(adj_cost)

    return render_template('op.html',
                           pred=f"Cost: ₹{cost:.2f}",
                           premium=f"Premium: ₹{premium}",
                           decision=f"Decision: {decision}",
                           claim=f"Claim: ₹{claim}")


# -------- CSV UPLOAD --------
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']
    df = pd.read_csv(file)

    df = clean_data(df)

    results = []

    for _, row in df.iterrows():

        bmi = calculate_bmi(row['height'], row['weight'])

        final = np.array([[
            row['age'],
            encode_gender(row['gender']),
            bmi,
            row['children'],
            encode_smoker(row['smoker']),
            encode_region(row['region'])
        ]])

        cost = model.predict(final)[0]
        adj_cost = adjust_risk(cost, row)

        results.append({
            "Predicted Cost": round(cost,2),
            "Premium": calculate_premium(adj_cost),
            "Decision": underwriting(adj_cost),
            "Claim": calculate_claim(adj_cost)
        })

    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    os.makedirs("static", exist_ok=True)
    result_df.to_csv("static/output.csv", index=False)

    return render_template('op.html',
                           tables=[result_df.to_html()],
                           download=True)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)