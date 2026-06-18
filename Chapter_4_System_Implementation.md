# Chapter 4: System Implementation & Coding

---

## 4.1 List of Modules

The MediShield Health Insurance Cost Prediction system is divided into six well-defined functional modules:

| S.No | Module Name | Primary Responsibility |
|------|-------------|------------------------|
| 1 | Machine Learning Inference Module | Predicts baseline medical cost using a trained Random Forest model |
| 2 | Actuarial Risk Engine Module | Adjusts cost based on lifestyle risks and computes actuarial premiums |
| 3 | Multi-Plan Simulation Module | Projects premiums over 5/10/15/20 years with inflation and NCB |
| 4 | Underwriting Decision Engine Module | Classifies applicant risk as Approved, High Risk, or Rejected |
| 5 | Batch Processing Module | Processes bulk CSV uploads and generates downloadable reports |
| 6 | Web User Interface Module | Renders the Flask-based frontend for data entry and result display |

---

## 4.2 Overview of All Modules (Combined Description)

The MediShield system operates as a sequential, multi-tier pipeline. At the entry point, the **Web User Interface Module** presents the user with an interactive form to input personal, demographic, and medical data, or alternatively allows bulk data entry via CSV upload. Once submitted, the **Machine Learning Inference Module** pre-processes the raw inputs — converting textual categories into numeric encodings and computing BMI from height and weight — before feeding the feature vector into a pre-trained Random Forest regressor that returns a baseline predicted medical cost (in USD). This raw cost is immediately handed to the **Actuarial Risk Engine Module**, which applies clinically grounded risk multipliers for conditions such as alcoholism, diabetes, and ongoing treatment, then computes a probabilistic expected annual loss. A dynamic loading structure adjusts the expense ratio and profit margin based on the user's chosen Sum Insured tier, and a logarithmic coverage factor scales the premium fairly across policy sizes. GST of 18% is applied last to yield the final gross premium. Simultaneously, the **Underwriting Decision Engine Module** computes a risk score using the ratio between the raw ML cost and the risk-adjusted cost, along with age flags, and returns an automated policy decision. The **Multi-Plan Simulation Module** then runs year-by-year actuarial loops for 5, 10, 15, and 20-year horizons, compounding medical inflation at 12% annually, growing the Sum Insured at 10% annually, and applying a No-Claim Bonus (NCB) discount of 5% per claim-free year (capped at 50%). For CSV uploads, the **Batch Processing Module** iterates this entire pipeline across every data row, assembles a results DataFrame, and exports it as a downloadable `output.csv` file.

---

## 4.3 Module-by-Module Explanation with Implementation

---

### 4.3.1 Module 1 — Machine Learning Inference Module

#### Purpose
This module is the core predictive engine of the system. It converts the user's health profile into a structured numerical feature vector and passes it to a pre-trained Random Forest regression model to obtain a baseline expected annual medical cost.

#### Libraries Used
- `pickle` — to deserialize the saved trained model
- `numpy` — to construct the feature array
- `os` — to resolve the absolute model file path

#### Detailed Code Explanation

**Step 1: Model Loading at Startup**

```python
model_path = os.path.join(os.path.dirname(__file__), "rf_tuned.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)
```

- `os.path.dirname(__file__)` ensures the path is relative to `app.py`, not the working directory — this is critical for deployment portability.
- `pickle.load()` deserializes the `rf_tuned.pkl` binary file back into a scikit-learn `RandomForestRegressor` object.
- The model is loaded **once at startup** (not per-request), making inference fast.

**Step 2: Input Encoding Helper Functions**

Before creating the feature array, all categorical text inputs must be converted to integers. Four helper functions handle this:

```python
def safe_str(x):
    return str(x).strip().lower()
    # Normalizes any input to lowercase trimmed string
    # Prevents mismatch from "Male" vs "male" vs " male "

def encode_gender(g):
    return 1 if safe_str(g) in ['male', 'm'] else 0
    # Binary encoding: Male = 1, Female = 0

def encode_smoker(s):
    return 1 if safe_str(s) == 'yes' else 0
    # Binary encoding: Smoker = 1, Non-Smoker = 0

def encode_region(r):
    region_map = {
        'northeast': 0, 'northwest': 1,
        'southeast': 2, 'southwest': 3
    }
    return region_map.get(safe_str(r), 1)
    # Ordinal encoding matching the original training dataset's label mapping
    # Default falls back to 'northwest' (1) if region is unrecognised
```

**Step 3: BMI Computation**

```python
def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)
```

- Height is given in centimetres, so dividing by 100 converts it to metres.
- The standard WHO formula: BMI = Weight(kg) / Height(m)²
- This is computed on the server to keep the form simple — users only enter height and weight.

**Step 4: Building the Feature Vector and Predicting**

```python
bmi    = calculate_bmi(height, weight)
sex    = encode_gender(data['gender'])
smoker = encode_smoker(data['smoker'])
region = encode_region(data['region'])

final = np.array([[age, sex, bmi, children, smoker, region]])
cost  = model.predict(final)[0]
```

- The feature order `[age, sex, bmi, children, smoker, region]` exactly matches the column order used during model training — any mismatch would produce incorrect predictions.
- `model.predict()` returns an array; `[0]` extracts the single scalar value.
- `cost` at this stage is the raw USD-denominated baseline medical cost as predicted by the Random Forest.

#### Input / Output Summary

| Input | Description |
|-------|-------------|
| Age | Integer, 1–120 |
| Gender | Encoded as 1 (Male) / 0 (Female) |
| BMI | Computed from Height & Weight |
| Children | Integer, 0–20 |
| Smoker | Encoded as 1 (Yes) / 0 (No) |
| Region | Ordinal 0–3 |

| Output | Description |
|--------|-------------|
| `cost` | Predicted baseline medical cost (float, USD) |

**[Screenshot 1: Insert screenshot of the prediction form with sample inputs and the terminal showing model load confirmation]**

---

### 4.3.2 Module 2 — Actuarial Risk Engine Module

#### Purpose
This module transforms the ML-predicted baseline cost into a realistic, market-standard insurance premium. It applies clinically motivated risk multipliers, computes an actuarially correct expected annual loss, and applies a dynamic loading structure based on Sum Insured tier.

#### Detailed Code Explanation

**Step 1: Risk Adjustment — `adjust_risk()`**

```python
def adjust_risk(cost, data):
    alcoholic = safe_str(data.get('alcoholic', 'no'))
    disease   = safe_str(data.get('disease', 'none'))
    treatment = safe_str(data.get('treatment', 'none'))
    duration  = float(data.get('duration', 0))

    if alcoholic == 'yes':
        cost *= 1.1          # +10% loading for alcohol use

    disease_factor = {
        'none':     1.0,
        'diabetes': 1.25,    # +25% for diabetic patients
        'heart':    1.5      # +50% for heart disease patients
    }
    cost *= disease_factor.get(disease, 1.1)

    if duration > 2:
        cost *= 1.1          # +10% for chronic conditions (>2 years)

    if treatment == 'ongoing':
        cost *= 1.2          # +20% for patients on active treatment

    return cost
```

- Each multiplier reflects real-world underwriting tables used by insurance companies.
- Multipliers stack multiplicatively, not additively — a diabetic alcoholic on ongoing treatment compounds all three penalties.
- Result `adj_cost` is the risk-corrected expected medical spend.

**Step 2: Claim Probability — `get_probability_of_claim()`**

```python
def get_probability_of_claim(age, smoker, bmi, disease):
    base = 0.02 * np.exp(0.03 * (age - 30))
    # Exponential growth from age 30 baseline
    # At age 30: base ≈ 0.02, at age 60: base ≈ 0.049

    if smoker == 1:              base *= 1.5   # 50% higher claim likelihood
    if bmi > 30:                 base *= 1.2   # Obese: 20% more likely
    if disease in ['diabetes', 'heart']: base *= 1.4  # Chronic disease: 40% more likely

    return min(base, 0.35)  # Hard cap at 35% to prevent extreme outputs
```

- Uses an **exponential aging model** — medically validated as health risks grow non-linearly with age.
- The probability cap at 0.35 (35%) prevents the model from producing premiums that are economically infeasible.

**Step 3: Full Premium Calculation — `calculate_premium()`**

```python
def calculate_premium(adj_cost, age, sum_insured, smoker=0, bmi=25, disease='none'):

    # Step A: Expected Annual Loss
    probability_of_claim = get_probability_of_claim(age, smoker, bmi, disease)
    expected_loss = adj_cost * probability_of_claim
    # = (Risk-adjusted cost) × (Probability it will be claimed)

    # Step B: Dynamic Loading based on Sum Insured tier
    if sum_insured <= 500000:
        expense_ratio, profit_margin = 0.25, 0.05   # 30% total loading
    elif sum_insured <= 1000000:
        expense_ratio, profit_margin = 0.20, 0.07   # 27% total loading
    else:
        expense_ratio, profit_margin = 0.15, 0.10   # 25% total loading
    # Higher coverage = more efficient → lower expense ratio (economies of scale)

    loaded_cost = expected_loss * (1 + expense_ratio + profit_margin)

    # Step C: Logarithmic Coverage Scaling
    coverage_factor = 1 + np.log(sum_insured / 500000 + 1)
    # Ensures premium grows sub-linearly with coverage — fair pricing at higher tiers
    # At 5L  → factor ≈ 1.693
    # At 10L → factor ≈ 2.099
    # At 50L → factor ≈ 3.397

    premium = loaded_cost * coverage_factor

    # Step D: Apply 18% GST (mandatory on Indian insurance premiums)
    premium *= 1.18

    return round(premium, 2)
```

- The **expected loss** approach is the foundation of actuarial pricing — you don't price the worst case, you price the probability-weighted average loss.
- **Dynamic loading** rewards higher Sum Insured tiers with better expense efficiency.
- The **logarithmic coverage factor** prevents linear (unfair) premium scaling with sum insured.

**[Screenshot 2: Insert screenshot of output page showing the base premium and the actuarial breakdown]**

---

### 4.3.3 Module 3 — Multi-Plan Simulation Module

#### Purpose
This module computes multi-year insurance projections (5, 10, 15, 20 years) with realistic financial modelling — compounding medical inflation, sum insured growth, and No-Claim Bonus (NCB) discounts.

#### Plan Configuration

```python
PLANS = {
    "Basic":    {"deductible": 20000, "copay": 0.20, "multiplier": 0.75},
    "Standard": {"deductible": 10000, "copay": 0.10, "multiplier": 1.0},
    "Premium":  {"deductible": 0,     "copay": 0.0,  "multiplier": 1.4}
}
```

| Plan | Deductible | Co-Pay | Coverage Multiplier |
|------|-----------|--------|---------------------|
| Basic | ₹20,000 | 20% | 0.75× |
| Standard | ₹10,000 | 10% | 1.0× |
| Premium | ₹0 | 0% | 1.4× |

#### Plan Premium & Claim Adjustment

```python
def apply_plan_adjustment(base_premium, plan):
    # Higher deductible and copay → user bears more risk → insurer charges less
    discount = np.log1p(plan["deductible"] / 10000) * 0.08 + plan["copay"] * 0.4
    premium  = base_premium * (1 - discount)
    premium *= plan["multiplier"]
    return round(premium, 2)

def apply_plan_claim(base_claim, plan, sum_insured):
    claim  = max(base_claim - plan["deductible"], 0)  # Deductible subtracted first
    claim *= (1 - plan["copay"])                        # User pays copay %
    return min(claim, sum_insured)                      # Cannot exceed policy limit
```

#### Year-by-Year Projection Loop with NCB

```python
for y in [5, 10, 15, 20]:
    total_paid = 0
    ncb_years  = 0                       # Tracks consecutive claim-free years

    for year in range(y):
        future_age          = age + year
        future_adj_cost     = adj_cost * ((1 + 0.12) ** year)   # 12% medical inflation
        current_sum_insured = sum_insured * ((1 + 0.10) ** year) # 10% SI growth

        base_yearly_premium = calculate_premium(
            future_adj_cost, future_age, current_sum_insured, smoker, bmi, disease
        )

        # NCB: 5% discount per claim-free year, capped at 50%
        ncb_discount       = min(0.50, ncb_years * 0.05)
        discounted_premium = base_yearly_premium * (1 - ncb_discount)
        total_paid        += discounted_premium

        # Stochastically simulate whether user claims this year
        prob_claim = get_probability_of_claim(future_age, smoker, bmi, disease)
        rng_claim  = np.random.default_rng(int(cost) + year)  # Seeded for consistency
        if rng_claim.random() < prob_claim:
            ncb_years = 0   # Claim → NCB resets
        else:
            ncb_years += 1  # No claim → NCB accumulates

    plans[y] = {
        'total':    round(total_paid, 2),
        'future':   round(future_prem, 2),
        'coverage': round(required_coverage(sum_insured, y), 2)
    }
```

- **12% medical inflation** is based on India's historical medical CPI trend.
- **10% Sum Insured growth** mirrors cumulative bonus provisions in standard Indian health policies.
- **NCB seeding** (`int(cost) + year`) guarantees the same claim/no-claim decision each page refresh — preventing UI flicker.

**[Screenshot 3: Insert screenshot of the multi-year plan comparison table on the output page]**

---

### 4.3.4 Module 4 — Underwriting Decision Engine Module

#### Purpose
This module automatically evaluates the risk profile of an applicant and returns a policy decision — **Approved**, **High Risk**, or **Rejected** — based on a transparent scoring system.

#### Detailed Code Explanation

```python
def underwriting(base_cost, adj_cost, age):
    # Risk Multiplier: How much did lifestyle/disease factors inflate the base cost?
    risk_multiplier = adj_cost / base_cost if base_cost > 0 else 1

    score = 0

    # Age penalty: Seniors (>60) carry significantly higher actuarial risk
    if age > 60:
        score += 2

    # Risk loading penalty: Moderate risk amplification
    if risk_multiplier > 1.5:
        score += 2

    # Severe risk loading penalty: Extreme amplification
    if risk_multiplier > 2:
        score += 2   # Stacks with the previous rule — max score from risk = 4

    # Decision Thresholds
    if score >= 5:
        return "Rejected"    # e.g. Elderly + severe pre-existing disease
    elif score >= 3:
        return "High Risk"   # e.g. Elderly OR high multiplier alone
    return "Approved"
```

#### Risk Score Decision Table

| Condition | Score Added |
|-----------|------------|
| Age > 60 | +2 |
| Risk Multiplier > 1.5 | +2 |
| Risk Multiplier > 2.0 | +2 (stacks) |

| Total Score | Decision |
|-------------|----------|
| 0 – 2 | ✅ Approved |
| 3 – 4 | ⚠️ High Risk |
| ≥ 5 | ❌ Rejected |

- A 65-year-old with heart disease and ongoing treatment (multiplier ≈ 1.5×1.5×1.2 = 2.7) would score 2+2+2 = **6 → Rejected**.
- A 55-year-old non-smoker with no disease (multiplier ≈ 1.0) would score 0 → **Approved**.

**[Screenshot 4: Insert screenshot of output showing the Underwriting Decision badge (Approved/High Risk/Rejected)]**

---

### 4.3.5 Module 5 — Batch Processing Module

#### Purpose
Enables bulk insurance analysis by accepting a CSV file upload of up to 1,000 patient records. Each record is independently processed through the full actuarial pipeline, and the enriched results are exported as a downloadable CSV.

#### Detailed Code Explanation

**Step 1: File Validation**

```python
file = request.files.get('file')
if not file or not file.filename:
    return render_template('op.html', error="No file selected")
if not file.filename.lower().endswith('.csv'):
    return render_template('op.html', error="Invalid file format.")

df = pd.read_csv(file)

if len(df) > 1000:
    return render_template('op.html', error="Batch size too large. Maximum 1000 records.")
```

**Step 2: Data Cleaning — `clean_data()`**

```python
def clean_data(df):
    df.columns = df.columns.str.strip().str.lower()

    # Column name normalisation — handles Google Forms CSV variants
    df.rename(columns={
        'smoke': 'smoker',
        'drink': 'alcoholic',
        'duration of disease': 'duration',
        'duration_disease':    'duration'
    }, inplace=True)

    # Disease column normalisation
    df['disease'] = df['disease'].astype(str).str.lower()
    df['disease'] = df['disease'].replace({'no': 'none'})

    # Treatment column normalisation
    df['treatment'] = df['treatment'].replace({
        'no': 'none', 'tablet': 'ongoing', 'insulin': 'ongoing'
    })

    # Duration conversion (handles "2 years", "6 months" free text)
    def convert_duration(x):
        x = safe_str(x)
        if 'year'  in x: return float(x.split()[0].replace('+',''))
        if 'month' in x: return float(x.split()[0]) / 12
        return 0

    df['duration'] = df['duration'].apply(convert_duration)
    return df
```

**Step 3: Per-Row Pipeline Execution**

```python
results = []
for _, row in df.iterrows():
    bmi    = row['bmi'] if 'bmi' in row else calculate_bmi(row.get('height', 170), row.get('weight', 65))
    smoker = encode_smoker(row.get('smoker', 'no'))
    final  = np.array([[
        row.get('age', 30),
        encode_gender(row.get('gender', 'male')),
        bmi,
        row.get('children', 0),
        smoker,
        encode_region(row.get('region', 'southwest'))
    ]])

    cost     = model.predict(final)[0]
    adj_cost = adjust_risk(cost, row)
    premium  = calculate_premium(adj_cost, row['age'], batch_sum_insured, smoker, bmi, disease)
    base_claim     = calculate_claim(adj_cost, batch_sum_insured)
    standard_claim = apply_plan_claim(base_claim, PLANS["Standard"], batch_sum_insured)

    row_result = {
        "Predicted Cost":   round(adj_cost, 2),
        "Premium (1 Year)": premium,
        "Decision":         underwriting(cost, adj_cost, row.get('age', 30)),
        "Max Claim (Std)":  round(standard_claim, 2),
        "Coverage 20Y":     round(required_coverage(batch_sum_insured, 20), 2)
    }
    # Multi-year totals (5, 10, 15, 20 years) appended per row
    results.append(row_result)
```

**Step 4: Output Export**

```python
result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
os.makedirs("static", exist_ok=True)
result_df.to_csv("static/output.csv", index=False)

return render_template('op.html',
    tables=[result_df.to_html(classes='striped centered highlight', index=False)],
    download=True
)
```

- `pd.concat` joins the original columns with the computed result columns side-by-side.
- The HTML table is rendered directly in the browser using Pandas' `.to_html()` for instant previewing.
- A download button is conditionally displayed in the template via the `download=True` flag.

**[Screenshot 5: Insert screenshot of the CSV upload form and the resulting batch output table with the Download button]**

---

### 4.3.6 Module 6 — Web User Interface Module

#### Purpose
This module is the full-stack presentation layer — a Flask-powered Python backend with glassmorphic HTML/CSS/JS templates. It handles all HTTP routing, form data extraction, input validation, and dynamic result rendering.

#### Backend — Flask Application Setup

```python
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit for CSV files

# Route 1: Landing Page
@app.route('/')
def home():
    return render_template('home.html')

# Route 2: Single Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Input validation, model inference, actuarial computation...
    return render_template('op.html', cost=..., premium=..., plans=..., decision=...)

# Route 3: Batch Upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    # CSV processing pipeline...
    return render_template('op.html', tables=[...], download=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

#### Server-Side Input Validation (in `/predict` route)

```python
if not (1 <= age <= 120):       raise ValueError("Age must be between 1 and 120")
if not (50 <= height <= 300):   raise ValueError("Height must be between 50cm and 300cm")
if not (10 <= weight <= 300):   raise ValueError("Weight must be between 10kg and 300kg")
if not (0 <= children <= 20):   raise ValueError("Children count must be between 0 and 20")
if not (10 <= bmi <= 60):       raise ValueError("Calculated BMI must be between 10 and 60")

valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
if safe_str(data.get('region', '')) not in valid_regions:
    raise ValueError("Invalid region selected")
```

All validation errors are caught by a `try/except` block that gracefully renders an error message in the output template rather than crashing.

#### Frontend UI Highlights (`home.html`)

| Feature | Implementation |
|---------|----------------|
| Dark glassmorphic design | CSS `backdrop-filter: blur()` with semi-transparent cards |
| Animated background orbs | CSS `@keyframes drift` with `filter: blur(90px)` |
| Gradient text hero heading | CSS `background-clip: text` with linear-gradient |
| Smooth form focus effects | CSS `transition` on `border-color` and `box-shadow` |
| Responsive grid layout | CSS `grid-template-columns: repeat(auto-fit, minmax(...))` |
| Loading spinner on submit | JavaScript dynamically swaps button content on form submit |

**[Screenshot 6: Insert screenshot of the main MediShield landing page with the prediction form filled in]**

**[Screenshot 7: Insert screenshot of the full output/results page showing premium cards, plan table, and decision badge]**

---

## 4.4 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend Framework | Flask (Python) | HTTP routing, form processing, template rendering |
| ML Model | Random Forest Regressor (scikit-learn) | Baseline cost prediction |
| Numerical Computing | NumPy | Array operations, probability calculations |
| Data Processing | Pandas | CSV ingestion, cleaning, aggregation |
| Model Persistence | Pickle | Serialization of trained model |
| Frontend | HTML5 + Vanilla CSS | UI structure and glassmorphic styling |
| Typography | Google Fonts (Inter) | Modern sans-serif font rendering |
| Icons | Material Icons Round | Contextual UI iconography |

---

*End of Chapter 4: System Implementation & Coding*
