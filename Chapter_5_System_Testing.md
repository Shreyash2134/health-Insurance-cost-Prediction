# Chapter 5: System Testing

The testing phase of the MediShield Health Insurance Cost Prediction project is critical to ensuring the model's accuracy, robustness, and reliability in real-world applications. Initially, a comprehensive evaluation is conducted using a separate validation dataset to measure performance metrics such as Mean Absolute Error (MAE), R-squared (R²), and overall regression accuracy. These metrics provide insights into the model's capability to accurately predict medical insurance costs and distinguish them from non-risk-adjusted statistical averages.

Testing is a structured phase of the software development life cycle that verifies whether the developed application satisfies its defined requirements. For the MediShield system, a multi-level testing strategy was adopted covering Unit Testing, Functionality Testing, and Integration Testing. Each level served a distinct purpose — from validating individual mathematical functions in isolation to verifying that the complete end-to-end workflow operates correctly under real-world usage conditions.

---

## 5.1 UNIT TESTING

Unit testing was performed to verify the correctness of individual functions and logic components of the MediShield system in isolation, ensuring that each unit of code behaves as expected before being combined with other modules.

The **Input Encoding Functions** were tested to confirm that categorical text values submitted by the user are correctly converted into numerical values required by the Machine Learning model. The `encode_gender()` function was verified to return `1` for inputs like `"male"` and `"Male"`, and `0` for `"female"` or any unrecognized value. The `encode_smoker()` function was confirmed to return `1` only when the input is `"yes"` (case-insensitive) and `0` for all other values. The `encode_region()` function was tested against all four valid region names and an invalid input, confirming it correctly defaults to Northwest (value `1`) when an unrecognized region is provided. The `safe_str()` utility was independently verified to handle uppercase letters, trailing spaces, and integer inputs — all of which are normalized to a clean, lowercase string before further processing.

The **BMI Calculation** function was tested with multiple height and weight combinations to confirm the standard WHO formula is correctly implemented. A person with height 170 cm and weight 70 kg was verified to produce a BMI of 24.22. A test case with height 160 cm and weight 100 kg correctly returned a BMI above 30, classifying the subject as obese. An edge case with height 100 cm and weight 25 kg returned exactly 25.0, confirming the centimetre-to-metre unit conversion is handled accurately.

The **Risk Adjustment Logic** function `adjust_risk()` was tested with a base cost of ₹10,000 and various risk combinations. Supplying `alcoholic = "yes"` correctly inflated the cost to ₹11,000. Setting `disease = "heart"` raised the cost to ₹15,000. When all risk factors were simultaneously active — alcoholism, heart disease, ongoing treatment, and a disease duration greater than two years — the function correctly stacked all multipliers in sequence, producing a compounded output that reflects cumulative clinical risk.

The **Claim Probability** function `get_probability_of_claim()` was tested by comparing a young, healthy, non-smoking individual against progressively riskier profiles. The results confirmed that claim probability grows consistently with age, smoking status, obesity (BMI above 30), and chronic disease. Crucially, an extreme high-risk profile — a 90-year-old smoker with a BMI of 45 and heart disease — was tested to confirm that the probability is correctly capped at 0.35 (35%), preventing economically infeasible premium outputs.

The **Premium Calculation** function was tested to verify that a higher Sum Insured always produces a higher premium, that smokers consistently pay more than non-smokers with identical other attributes, and that the final result is always returned rounded to exactly two decimal places. The **Underwriting Decision** function was tested with three carefully constructed profiles. A 30-year-old with no risk factors correctly returned `"Approved"`. A 65-year-old with a risk multiplier of 1.8 correctly returned `"High Risk"`. A 65-year-old with a multiplier of 2.5 correctly returned `"Rejected"`. A zero base cost was also tested to confirm the system defaults to a safe multiplier of 1 rather than throwing a division-by-zero error.

The **Plan Adjustment and Claim Computation** functions were tested against all three plan tiers — Basic, Standard, and Premium. The tests confirmed the expected premium ordering: the Basic plan yields the lowest premium due to its high deductible and co-pay, while the Premium plan yields the highest. A claim of ₹30,000 submitted to the Basic plan (₹20,000 deductible, 20% co-pay) correctly returned ₹8,000 after applying both deductions. A claim below the deductible — ₹15,000 against a ₹20,000 deductible — correctly returned zero, confirming the policyholder bears the full cost in such a scenario.

Overall, all unit tests passed successfully, confirming that the individual mathematical and logical components of the MediShield system operate correctly in isolation.

---

## 5.2 FUNCTIONALITY TESTING

Functionality testing checks that all the features of the proposed MediShield Health Insurance Cost Prediction system behave as expected under normal usage conditions. The testing was carried out on the currently implemented modules of the system to ensure proper functionality and user interaction.

The **Health Profile Submission and Form Validation** module was tested to ensure that users can enter their demographic and medical details via the interactive form. When a complete and valid profile was submitted — with age 35, height 170 cm, weight 70 kg, male gender, non-smoker status, Southwest region, and a ₹5 Lakh coverage goal — the system successfully processed the request and rendered the results page with premium calculations and plan comparisons. The input validation layer was then tested by deliberately entering out-of-range values. When the age field was set to 200, the system returned the error "Age must be between 1 and 120" and did not proceed with the prediction. When height and weight values that produce an extreme BMI were entered, the system correctly intercepted the request with the error "Calculated BMI must be between 10 and 60". An invalid region value was tested and similarly caught, confirming that the validation layer reliably guards against erroneous inputs before they reach the model.

The **Insurance Prediction and Premium Display** module was tested to verify that all key outputs are correctly displayed on the results page. The Predicted Medical Cost, Annual Premium, and the three plan tier cards (Basic, Standard, Premium) were each verified to be present and logically ordered — Basic being the cheapest and Premium the most expensive. The multi-year projection table was examined to confirm that 5, 10, 15, and 20-year total premium estimates and projected coverage values are computed and displayed accurately. The No-Claim Bonus effect was also verified indirectly by confirming that a younger, lower-risk profile yielded a noticeably lower future premium projection compared to an older high-risk profile over the same time horizon.

The **Underwriting Decision Engine** was functionally tested by submitting profiles specifically designed to trigger each of the three decision outcomes. For the Approved status, a 32-year-old non-smoking female with no pre-existing conditions was submitted, and the system correctly displayed the Approved badge. For High Risk, a 62-year-old patient with diabetes was submitted, and the system correctly flagged the application accordingly. For Rejected, a 68-year-old smoker with heart disease on ongoing treatment was submitted, and the system correctly displayed the Rejected badge, confirming that the decision engine output is accurately communicated through the frontend template.

The **Batch CSV Upload and Processing** module was tested using several CSV files with varying characteristics. A valid CSV file containing four patient records was uploaded, and the system correctly processed all rows and displayed a formatted HTML results table with predicted costs, premiums, decisions, and multi-year totals. A Google Forms-style CSV with non-standard column names such as `"smoke"`, `"drink"`, and `"duration of disease"` was also uploaded, and the system successfully normalised these headers and processed the data without errors. An attempt to upload a `.txt` file was correctly blocked with the error "Invalid file format". A batch file with 1,001 rows was rejected with the message "Batch size too large. Maximum 1000 records allowed per upload." After a successful batch upload, the download link was verified and the downloaded `output.csv` confirmed to contain all original and computed columns intact.

Overall, functionality testing confirmed that all implemented features of the MediShield system operate correctly from the user's perspective, handling both valid inputs and error conditions appropriately.

---

## 5.3 INTEGRATION TESTING

Integration testing was performed to verify the interaction between the different modules of the MediShield system and to ensure proper data flow between the frontend, backend, machine learning model, actuarial engine, and file storage components.

The **Form Submission and Backend Integration** was tested to ensure that data entered by the user on the `home.html` page is correctly transmitted to the Flask `/predict` route. When the form is submitted, the browser sends a POST request carrying all field values. The test verified that Flask's `request.form` dictionary correctly receives and stores every field — including `age`, `gender`, `smoker`, `region`, `disease`, `treatment`, `duration`, and `sum_insured` — without any data being lost or mistyped in transit. This confirmed proper linkage between the frontend interface and the backend processing logic.

The **Prediction Form and Machine Learning Model Integration** was tested to ensure that the encoded feature vector assembled from the form data is correctly passed to the Random Forest model. The test verified that the feature order `[age, sex, bmi, children, smoker, region]` exactly matches the column order used during training, and that `model.predict()` returns a valid float value without type errors. Predictions for a smoker and a non-smoker with otherwise identical profiles were compared, confirming that the model produces distinct and logically ordered outputs.

The **Machine Learning Output and Actuarial Engine Integration** was tested to confirm that the raw predicted cost returned by the ML model is immediately consumable by the `adjust_risk()` and `calculate_premium()` functions. The test traced a smoker's predicted cost from the ML model through the risk adjustment function and into the premium calculator, verifying that both the model's higher base prediction and the actuarial engine's claim probability amplification are applied in sequence, producing a substantially higher premium than for a non-smoker with the same profile.

The **Batch Processing Pipeline Integration** was tested end-to-end. Starting from a raw CSV upload, the data was tracked through the `clean_data()` normalisation function, through the row-by-row prediction loop, through the actuarial calculations, and into the final `pd.concat()` merge step. The test confirmed that the resulting DataFrame contains exactly as many rows as the original file and includes all expected computed columns — Predicted Cost, Premium, Decision, Max Claim, and multi-year totals. The file was then verified to be correctly written to `static/output.csv` and made accessible through the download link on the results page.

The **Frontend and Results Page Integration** was tested to confirm that all computed variables passed by the Flask route are correctly rendered by the Jinja2 template engine. The template was verified to correctly display the premium value, the plan comparison cards, the underwriting decision badge, and the multi-year projection data — all sourced from dynamically injected template variables without any rendering errors.

The **Environment and Server Integration** was tested to confirm proper functioning of the application within the local development environment. The Flask application was launched using `python app.py` on `localhost:5000`, and both the prediction form and the CSV upload were tested from a web browser on the same machine. The system successfully processed requests end-to-end within this environment, and all static files, templates, and the machine learning model were resolved correctly using relative path configurations, ensuring the system functions reliably without dependency on any specific directory structure.

Overall, integration testing confirms that all modules of the MediShield system — including the frontend, backend, machine learning model, actuarial engine, and file storage — interact correctly and perform coordinated operations for smooth and accurate prediction functionality.

---

## 5.4 TEST CASE SUMMARY TABLE

| Test ID | Type | Feature Tested | Input | Expected Output | Status |
| :--- | :---: | :--- | :--- | :--- | :---: |
| TC-U01 | Unit | `calculate_bmi` | H:170cm, W:70kg | 24.22 | ✅ Pass |
| TC-U02 | Unit | `encode_gender` | "Male" | 1 | ✅ Pass |
| TC-U03 | Unit | `encode_smoker` | "Yes" | 1 | ✅ Pass |
| TC-U04 | Unit | `adjust_risk` | Cost: ₹10,000, Alcoholic: Yes | ₹11,000 | ✅ Pass |
| TC-U05 | Unit | `adjust_risk` | Cost: ₹10,000, Heart Disease | ₹15,000 | ✅ Pass |
| TC-U06 | Unit | `adjust_risk` | All risk factors combined | Stacked multiplicative result | ✅ Pass |
| TC-U07 | Unit | `get_probability_of_claim` | Age:90, Smoker, BMI:45, Heart | ≤ 0.35 (cap verified) | ✅ Pass |
| TC-U08 | Unit | `underwriting` | Age: 65, Multiplier: 2.5 | "Rejected" | ✅ Pass |
| TC-U09 | Unit | `underwriting` | Age: 65, Multiplier: 1.8 | "High Risk" | ✅ Pass |
| TC-U10 | Unit | `apply_plan_claim` | Claim: ₹15,000, Basic Plan | ₹0 (below deductible) | ✅ Pass |
| TC-U11 | Unit | `calculate_claim` | Cost: ₹50,000 (called twice) | Same result both times | ✅ Pass |
| TC-F01 | Functional | Form Validation | Age: 200 | Error: Age out of range | ✅ Pass |
| TC-F02 | Functional | Form Validation | Extreme height/weight (BMI ≈ 1.25) | Error: BMI out of range | ✅ Pass |
| TC-F03 | Functional | Single Prediction | Valid 35-yr male profile | Results page with premium + plans | ✅ Pass |
| TC-F04 | Functional | Underwriting Badge | 68-yr smoker with heart disease | "Rejected" badge displayed | ✅ Pass |
| TC-F05 | Functional | Multi-Year Plans | Any valid profile | 5/10/15/20 year tables shown | ✅ Pass |
| TC-F06 | Functional | CSV Upload | Valid 4-row CSV | Data table + download link shown | ✅ Pass |
| TC-F07 | Functional | CSV File Validation | .txt file upload | Error: Invalid file format | ✅ Pass |
| TC-F08 | Functional | Batch Size Limit | 1,001-row CSV | Error: Batch size too large | ✅ Pass |
| TC-F09 | Functional | Google Forms CSV | Non-standard column headers | Processed successfully after cleaning | ✅ Pass |
| TC-I01 | Integration | Form → `/predict` Route | Form POST data | All fields received by Flask correctly | ✅ Pass |
| TC-I02 | Integration | Features → ML Model | Encoded feature array | Valid prediction returned | ✅ Pass |
| TC-I03 | Integration | ML Output → Actuarial Engine | Base cost (float) | Risk-adjusted premium computed | ✅ Pass |
| TC-I04 | Integration | Smoker vs Non-Smoker Pipeline | Two identical profiles, smoker differs | Smoker produces higher premium | ✅ Pass |
| TC-I05 | Integration | Batch CSV → Result DataFrame | Multi-row CSV file | All computed columns present per row | ✅ Pass |
| TC-I06 | Integration | Result DataFrame → Download | `output.csv` written to `/static` | File downloadable from results page | ✅ Pass |
