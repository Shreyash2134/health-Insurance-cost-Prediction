# Health Insurance Cost Prediction & Underwriting System

This project is a comprehensive Health Insurance Cost Predictor and Underwriting Engine built using Python, Flask, and scikit-learn. It uses a machine learning model to estimate base medical costs based on demographic and physical attributes, and it applies actuarial logic to compute insurance premiums, simulate various coverage plans, and provide an automated underwriting decision.

## Overview

The application features two main components:
1. **Single Individual Prediction**: A web interface where users can enter their details (age, height, weight, smoking habits, region, pre-existing diseases, etc.) to get a detailed breakdown of their estimated medical costs, risk adjustments, predicted premiums for different plans, and an underwriting decision (Approved, High Risk, Rejected).
2. **Batch Prediction**: A feature to upload a CSV file containing multiple users' data. The application processes the records, predicts the costs and premiums for each individual, and allows you to download the summarized output as a CSV file.

## Features

- **Cost Prediction**: Utilizes a Random Forest Regressor trained on historical insurance data. Features used include Age, Sex, BMI, Children count, Smoker status, and Region.
- **Risk Adjustment**: Further adjusts the predicted medical cost based on lifestyle factors (e.g., alcohol consumption) and pre-existing diseases (e.g., diabetes, heart conditions).
- **Actuarial Premium Calculation**: Calculates premiums considering probability of claim, expected loss, expense margins, and coverage scaling. Includes GST computations.
- **Multi-Plan Simulation**: Simulates different coverage plans (Basic, Standard, Premium) by applying varying deductibles, co-pays, and multipliers to show users customized choices.
- **Underwriting Decision**: Automatically scores the risk based on age and adjusted cost multipliers to return decisions like "Approved", "High Risk", or "Rejected".
- **Future Projections**: Estimates premiums and required coverage amounts for 5, 10, 15, and 20 years considering inflation and No-Claim Bonuses (NCB).

## Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (Random Forest Regressor), pandas, numpy
- **Frontend**: HTML, CSS, JavaScript (Materialize CSS framework)
- **Deployment**: Configured with `Procfile` for platforms like Heroku

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreyash2134/health-Insurance-cost-Prediction.git
   cd health-Insurance-cost-Prediction
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   Make sure you have Flask, numpy, pandas, and scikit-learn installed.
   ```bash
   pip install Flask numpy pandas scikit-learn
   ```

4. **Run the Model Training (Optional):**
   The repository already includes a trained model (`rf_tuned.pkl`). If you wish to retrain it using `insurance.csv`, run:
   ```bash
   python train_model.py
   ```

5. **Start the Flask Application:**
   ```bash
   python app.py
   ```

6. **Access the App:**
   Open your web browser and navigate to `http://127.0.0.0:5000` (or `http://localhost:5000`).

## Usage

### Single Prediction
- Go to the homepage.
- Fill out the form with the required details: Age, Gender, Height, Weight, Children, Smoker status, Region, Sum Insured, and any pre-existing conditions.
- Click **Predict** to view the comprehensive cost breakdown, plan comparisons, and the underwriting decision.

### Batch Processing
- Navigate to the upload section.
- Upload a CSV file containing columns such as `age`, `gender`, `smoker`, `height`, `weight`, `region`, etc.
- The system will process up to 1000 records at a time and generate an output CSV file with predictions and decisions.

## Project Structure

- `app.py`: The main Flask application file handling routes, logic, and predictions.
- `train_model.py`: Script to train the Random Forest Regressor and save the model.
- `rf_tuned.pkl`: The serialized machine learning model.
- `insurance.csv`: The dataset used for training the model.
- `templates/`: Contains HTML files (`home.html`, `op.html`) for the web application UI.
- `static/`: Contains static assets like CSS (Materialize) and generated output CSV files.
- Diagram Files: Several `.drawio` files representing Activity, Class, Component, and Sequence diagrams for the project architecture.
- Markdown Files: Contains documentation chapters on feasibility, methodology, implementation, and testing.

## License
Feel free to use and modify the code as per your requirements.
