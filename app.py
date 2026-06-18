from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit

# ==============================
# LOAD MODEL
# ==============================
model_path = os.path.join(os.path.dirname(__file__), "rf_tuned.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# ==============================
# HELPER FUNCTIONS
# ==============================
def safe_str(x):
    return str(x).strip().lower()

def encode_gender(g):
    return 1 if safe_str(g) in ['male', 'm'] else 0

def encode_smoker(s):
    return 1 if safe_str(s) == 'yes' else 0

def encode_region(r):
    region_map = {
        'northeast': 0,
        'northwest': 1,
        'southeast': 2,
        'southwest': 3
    }
    return region_map.get(safe_str(r), 1)

def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)

# ==============================
# COVERAGE (INFLATION BASED)
# ==============================
def required_coverage(base_goal, years, inflation=0.12):
    return base_goal * ((1 + inflation) ** years)

# ==============================
# LIC LOGIC (IMPROVED)
# ==============================
def adjust_risk(cost, data):

    alcoholic = safe_str(data.get('alcoholic', 'no'))
    disease = safe_str(data.get('disease', 'none'))
    treatment = safe_str(data.get('treatment', 'none'))

    try:
        duration = float(data.get('duration', 0))
    except:
        duration = 0

    # Risk factors
    if alcoholic == 'yes':
        cost *= 1.1

    disease_factor = {
        'none': 1.0,
        'diabetes': 1.25,
        'heart': 1.5
    }
    cost *= disease_factor.get(disease, 1.1)

    if duration > 2:
        cost *= 1.1

    if treatment == 'ongoing':
        cost *= 1.2

    return cost


def get_probability_of_claim(age, smoker, bmi, disease):
    base = 0.02 * np.exp(0.03 * (age - 30))
    if smoker == 1: base *= 1.5
    if bmi > 30: base *= 1.2
    if disease in ['diabetes', 'heart']: base *= 1.4
    return min(base, 0.35)

def calculate_premium(adj_cost, age, sum_insured, smoker=0, bmi=25, disease='none'):
    # 1. Probability of Claim (Multi-factor)
    probability_of_claim = get_probability_of_claim(age, smoker, bmi, disease)

    # 2. Expected Annual Loss
    # adj_cost is the predicted bill size, expected_loss is probability-weighted risk
    expected_loss = adj_cost * probability_of_claim

    # 3. Expense + Profit Loading (DYNAMIC based on sum_insured)
    if sum_insured <= 500000:
        expense_ratio = 0.25
        profit_margin = 0.05
    elif sum_insured <= 1000000:
        expense_ratio = 0.20
        profit_margin = 0.07
    else:
        expense_ratio = 0.15
        profit_margin = 0.10
        
    loaded_cost = expected_loss * (1 + expense_ratio + profit_margin)

    # 4. Coverage Scaling (Better actuarial scaling)
    coverage_factor = 1 + np.log(sum_insured / 500000 + 1)
    
    premium = loaded_cost * coverage_factor

    # 5. GST (18%)
    premium *= 1.18

    return round(premium, 2)


def future_premium(base, years, rate=0.12):
    return round(base * ((1 + rate) ** years), 2)


def underwriting(base_cost, adj_cost, age):
    risk_multiplier = adj_cost / base_cost if base_cost > 0 else 1
    score = 0
    if age > 60: score += 2
    if risk_multiplier > 1.5: score += 2
    if risk_multiplier > 2: score += 2
    
    if score >= 5:
        return "Rejected"
    elif score >= 3:
        return "High Risk"
    return "Approved"


def calculate_claim(cost, sum_insured=500000):
    # Simulate a major medical event using a log-normal severity shock
    # Using cost to seed ensures consistent UI values across page refreshes
    rng = np.random.default_rng(int(cost))
    if rng.random() < 0.7:
        shock = rng.lognormal(mean=0.8, sigma=0.3)  # small claims
    else:
        shock = rng.lognormal(mean=1.8, sigma=0.6)  # large claims
    major_event_cost = cost * shock
    
    # Pure base claim (no copay/deductible here, handled by plan layer)
    return min(major_event_cost, sum_insured)

# ==============================
# MULTI-PLAN CONFIGURATION
# ==============================
PLANS = {
    "Basic": {
        "deductible": 20000,
        "copay": 0.20,
        "multiplier": 0.75
    },
    "Standard": {
        "deductible": 10000,
        "copay": 0.10,
        "multiplier": 1.0
    },
    "Premium": {
        "deductible": 0,
        "copay": 0.0,
        "multiplier": 1.4
    }
}

def apply_plan_adjustment(base_premium, plan):
    # Discount for higher deductible and copay (using log scaling for smooth deductible impact)
    discount = np.log1p(plan["deductible"] / 10000) * 0.08 + plan["copay"] * 0.4
    premium = base_premium * (1 - discount)
    premium *= plan["multiplier"]
    return round(premium, 2)

def apply_plan_claim(base_claim, plan, sum_insured):
    # Apply deductible
    claim = max(base_claim - plan["deductible"], 0)
    # Apply co-pay
    claim *= (1 - plan["copay"])
    return min(claim, sum_insured)

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

    if 'disease' not in df.columns:
        df['disease'] = 'none'
    df['disease'] = df['disease'].astype(str).str.lower()
    df['disease'] = df['disease'].replace({'no': 'none'})

    if 'treatment' not in df.columns:
        df['treatment'] = 'none'
    df['treatment'] = df['treatment'].astype(str).str.lower()
    df['treatment'] = df['treatment'].replace({
        'no': 'none',
        'tablet': 'ongoing',
        'insulin': 'ongoing'
    })

    def convert_duration(x):
        x = safe_str(x)
        try:
            if 'year' in x:
                return float(x.split()[0].replace('+',''))
            elif 'month' in x:
                return float(x.split()[0]) / 12
        except:
            return 0
        return 0

    if 'duration' not in df.columns:
        df['duration'] = 0
    else:
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

    try:
        age      = int(data['age'])
        height   = float(data['height'])
        weight   = float(data['weight'])
        children = int(data['children'])

        # Validation
        if not (1 <= age <= 120): raise ValueError("Age must be between 1 and 120")
        if not (50 <= height <= 300): raise ValueError("Height must be between 50cm and 300cm")
        if not (10 <= weight <= 300): raise ValueError("Weight must be between 10kg and 300kg")
        if not (0 <= children <= 20): raise ValueError("Children count must be between 0 and 20")

        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if safe_str(data.get('region', '')) not in valid_regions: raise ValueError("Invalid region selected")

        bmi    = calculate_bmi(height, weight)
        if not (10 <= bmi <= 60): raise ValueError("Calculated BMI must be between 10 and 60")
        sex    = encode_gender(data['gender'])
        smoker = encode_smoker(data['smoker'])
        region = encode_region(data['region'])

        final = np.array([[age, sex, bmi, children, smoker, region]])

        cost      = model.predict(final)[0]
        adj_cost  = adjust_risk(cost, data)

        # Read chosen Sum Insured from form, default to ₹5,00,000
        sum_insured = float(data.get('sum_insured', 500000))
        
        # Calculate Base Actuarial Premium
        base_premium = calculate_premium(adj_cost, age, sum_insured, smoker, bmi, data.get('disease', 'none'))
        
        # Calculate base claim using random shock severity
        base_claim   = calculate_claim(adj_cost, sum_insured)

        # Multi-Plan Output Generation
        plans_output = {}
        for name, plan in PLANS.items():
            plan_premium = apply_plan_adjustment(base_premium, plan)
            plan_claim   = apply_plan_claim(base_claim, plan, sum_insured)

            plans_output[name] = {
                "premium": plan_premium,
                "claim": round(plan_claim, 2),
                "deductible": plan["deductible"],
                "copay": int(plan["copay"] * 100)
            }
            
        yearly_premium = base_premium # For backward compatibility with 5/10/20 yr plans

        # All 4 coverage plans (Calculated Actuarially)
        plans = {}
        for y in [5, 10, 15, 20]:
            total_paid = 0
            ncb_years = 0
            for year in range(y):
                # Age increases by 1 each year
                future_age = age + year
                # Medical cost inflates by 12% each year
                future_adj_cost = adj_cost * ((1 + 0.12)**year)
                # Sum insured inflates by 10% each year (market standard super top-up / cumulative bonus)
                current_sum_insured = sum_insured * ((1 + 0.10)**year)
                
                # Actuarial premium for that specific year based on inflated sum insured
                base_yearly_premium = calculate_premium(future_adj_cost, future_age, current_sum_insured, smoker, bmi, data.get('disease', 'none'))
                
                # Apply No-Claim Bonus (NCB) discount: 5% per claim-free year, max 50%
                ncb_discount = min(0.50, ncb_years * 0.05)
                discounted_premium = base_yearly_premium * (1 - ncb_discount)
                
                total_paid += discounted_premium

                # Simulate claim probability to see if NCB resets for next year
                prob_claim = get_probability_of_claim(future_age, smoker, bmi, data.get('disease', 'none'))
                rng_claim = np.random.default_rng(int(cost) + year)
                if rng_claim.random() < prob_claim:
                    ncb_years = 0 # reset NCB on claim
                else:
                    ncb_years += 1
                
            # Future premium is the estimated premium for the final year
            final_yr_cost = adj_cost * ((1 + 0.12)**(y-1))
            final_yr_sum_insured = sum_insured * ((1 + 0.10)**(y-1))
            future_prem = calculate_premium(final_yr_cost, age + y - 1, final_yr_sum_insured, smoker, bmi, data.get('disease', 'none'))
            future_prem *= (1 - min(0.50, ncb_years * 0.05)) # Apply NCB to future displayed premium

            plans[y] = {
                'total':  round(total_paid, 2),
                'future': round(future_prem, 2),
                'coverage': round(required_coverage(sum_insured, y), 2)
            }

        decision = underwriting(cost, adj_cost, age)
        claim    = base_claim
        coverage = required_coverage(sum_insured, 20)

        return render_template('op.html',
            cost=round(adj_cost, 2),
            premium=yearly_premium,
            plans=plans,
            plans_output=plans_output,
            decision=decision,
            claim=claim,
            coverage=round(coverage, 2),
            sum_insured=sum_insured
        )

    except Exception as e:
        return render_template('op.html', error=str(e))


# -------- CSV UPLOAD --------
@app.route('/upload', methods=['POST'])
def upload():

    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return render_template('op.html', error="No file selected")
        if not file.filename.lower().endswith('.csv'):
            return render_template('op.html', error="Invalid file format. Please upload a valid CSV file.")

        try:
            df = pd.read_csv(file)
        except Exception:
            return render_template('op.html', error="Error reading CSV file. Make sure it is correctly formatted.")
            
        if len(df) > 1000:
            return render_template('op.html', error="Batch size too large. Maximum 1000 records allowed per upload.")

        # Clean data first to standardise column names like 'smoke' -> 'smoker'
        df = clean_data(df)

        # Ensure required base columns exist
        required_cols = ['age', 'gender', 'smoker']
        if not all(c in df.columns for c in required_cols):
            return render_template('op.html', error=f"CSV must contain at least: {', '.join(required_cols)}")
            
        # Data Sanitization
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age']).copy()
        df = df[(df['age'] >= 1) & (df['age'] <= 120)]
        
        batch_sum_insured = float(request.form.get('batch_sum_insured', 500000))

        results = []

        for _, row in df.iterrows():

            
            if 'bmi' in row:
                bmi = row['bmi']
            else:
                bmi = calculate_bmi(row.get('height', 170), row.get('weight', 65))

            smoker = encode_smoker(row.get('smoker', 'no'))
           
            final = np.array([[ 
                row.get('age', 30),
                encode_gender(row.get('gender', 'male')),
                bmi,
                row.get('children', 0),
                smoker,
                encode_region(row.get('region', 'southwest'))
            ]])

         
            cost = model.predict(final)[0]
            adj_cost = adjust_risk(cost, row)
            
            disease = row.get('disease', 'none')
            # Calculate Actuarial Premium (Using selected batch coverage)
            premium = calculate_premium(adj_cost, row.get('age', 30), batch_sum_insured, smoker, bmi, disease)
            
            # Standard plan payout simulation
            base_claim = calculate_claim(adj_cost, batch_sum_insured)
            standard_claim = apply_plan_claim(base_claim, PLANS["Standard"], batch_sum_insured)

            row_result = {
                "Predicted Cost":   round(adj_cost, 2),
                "Premium (1 Year)": premium,
                "Decision":         underwriting(cost, adj_cost, row.get('age', 30)),
                "Max Claim (Std)":  round(standard_claim, 2),
                "Coverage 20Y":     round(required_coverage(batch_sum_insured, 20), 2)
            }

            # All 4 plans (Dynamic actuarial pricing)
            for y in [5, 10, 15, 20]:
                total_paid = 0
                ncb_years = 0
                for year in range(y):
                    future_age = row.get('age', 30) + year
                    future_adj_cost = adj_cost * ((1 + 0.12)**year)
                    current_sum_insured = batch_sum_insured * ((1 + 0.10)**year)
                    
                    base_yearly_premium = calculate_premium(future_adj_cost, future_age, current_sum_insured, smoker, bmi, disease)
                    ncb_discount = min(0.50, ncb_years * 0.05)
                    total_paid += base_yearly_premium * (1 - ncb_discount)

                    prob_claim = get_probability_of_claim(future_age, smoker, bmi, disease)
                    rng_claim = np.random.default_rng(int(cost) + year)
                    if rng_claim.random() < prob_claim:
                        ncb_years = 0
                    else:
                        ncb_years += 1
                    
                row_result[f"Total {y}Y"]  = round(total_paid, 2)

            results.append(row_result)

        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)

        os.makedirs("static", exist_ok=True)
        result_df.to_csv("static/output.csv", index=False)

        return render_template('op.html',
            tables=[result_df.to_html(classes='striped centered highlight', index=False)],
            download=True
        )

    except Exception as e:
        return render_template('op.html', error=str(e))


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)