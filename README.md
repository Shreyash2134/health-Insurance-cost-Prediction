# 🏥 Health Insurance Cost Prediction & Premium Estimation

## 👨‍💻 Author
**Shreyash Kapse**  
RTMNU / SVPCET  
April 2026  

---

## 📌 Abstract
This project presents a machine learning-based system for predicting health insurance costs and estimating premiums using risk-based analysis. A Random Forest Regressor is trained on a healthcare dataset to estimate medical costs based on features such as age, BMI, smoking habits, and region. Additional risk factors like disease, alcohol consumption, and treatment duration are incorporated using rule-based adjustments inspired by real insurance underwriting practices. The system supports both individual predictions and CSV uploads. This project demonstrates how machine learning and actuarial concepts can be combined to simulate real-world insurance pricing.

---

## 📌 Introduction
Health insurance helps manage medical expenses and financial risk. Insurance companies calculate premiums based on risk factors such as age, health condition, and lifestyle. Traditionally, this is done using actuarial methods and underwriting.  

This project aims to simulate that process using machine learning by predicting medical cost and applying risk-based adjustments to estimate premium and claim.

---

## 📌 Literature Review
- Insurance pricing is based on **expected loss + risk loading**
- Regulatory bodies require **risk-based premium calculation**
- Machine learning is increasingly used for **risk prediction**

---

## ⚙️ Methodology
1. Train ML model (Random Forest) on insurance dataset  
2. Predict base medical cost  
3. Apply risk adjustments (disease, alcohol, duration, treatment)  
4. Calculate:
   - Premium  
   - Claim amount  
   - Approval decision  

---

## 🛠️ Implementation

### 🔹 Technologies Used
- Python  
- Flask  
- Scikit-learn  
- Pandas, NumPy  

### 🔹 Tools
- Jupyter Notebook  
- VS Code  
- GitHub  
- Google Forms (for data collection)  

---

## 📊 Results

### Example Output:
- **Predicted Cost:** ₹2,50,000  
- **Premium:** ₹15,000/year  
- **Claim:** ₹2,50,000  

### Observations:
- Smokers → higher cost  
- Diseases → increase risk  
- Higher BMI → higher prediction  

---

## ⚠️ Limitations
- Uses assumed risk multipliers  
- No real LIC data  
- Limited dataset  
- No waiting period / co-pay  

---

## 🚀 Future Scope
- Add real actuarial data  
- Multiple plans (₹2L / ₹5L / ₹10L)  
- Add waiting period logic  
- Improve ML model accuracy  
- Deploy as full web app  

---

## 📌 Conclusion
This project combines machine learning and risk-based logic to simulate insurance premium calculation. While real insurers use actuarial models, this system provides a simplified and practical approximation.

---

## 📚 References
1. Principles of Risk Management and Insurance – George E. Rejda  
2. IRDAI Health Insurance Regulations  
3. https://www.kaggle.com/datasets/mirichoi0218/insurance  
4. Actuarial Mathematics for Life Contingent Risks  

---

## 📁 Project Structure
# 🏥 Health Insurance Cost Prediction & Premium Estimation

## 👨‍💻 Author
**Shreyash Kapse**  
RTMNU / SVPCET  
April 2026  

---

## 📌 Abstract
This project presents a machine learning-based system for predicting health insurance costs and estimating premiums using risk-based analysis. A Random Forest Regressor is trained on a healthcare dataset to estimate medical costs based on features such as age, BMI, smoking habits, and region. Additional risk factors like disease, alcohol consumption, and treatment duration are incorporated using rule-based adjustments inspired by real insurance underwriting practices. The system supports both individual predictions and CSV uploads. This project demonstrates how machine learning and actuarial concepts can be combined to simulate real-world insurance pricing.

---

## 📌 Introduction
Health insurance helps manage medical expenses and financial risk. Insurance companies calculate premiums based on risk factors such as age, health condition, and lifestyle. Traditionally, this is done using actuarial methods and underwriting.  

This project aims to simulate that process using machine learning by predicting medical cost and applying risk-based adjustments to estimate premium and claim.

---

## 📌 Literature Review
- Insurance pricing is based on **expected loss + risk loading**
- Regulatory bodies require **risk-based premium calculation**
- Machine learning is increasingly used for **risk prediction**

---

## ⚙️ Methodology
1. Train ML model (Random Forest) on insurance dataset  
2. Predict base medical cost  
3. Apply risk adjustments (disease, alcohol, duration, treatment)  
4. Calculate:
   - Premium  
   - Claim amount  
   - Approval decision  

---

## 🛠️ Implementation

### 🔹 Technologies Used
- Python  
- Flask  
- Scikit-learn  
- Pandas, NumPy  

### 🔹 Tools
- Jupyter Notebook  
- VS Code  
- GitHub  
- Google Forms (for data collection)  

---

## 📊 Results

### Example Output:
- **Predicted Cost:** ₹2,50,000  
- **Premium:** ₹15,000/year  
- **Claim:** ₹2,50,000  

### Observations:
- Smokers → higher cost  
- Diseases → increase risk  
- Higher BMI → higher prediction  

---

## ⚠️ Limitations
- Uses assumed risk multipliers  
- No real LIC data  
- Limited dataset  
- No waiting period / co-pay  

---

## 🚀 Future Scope
- Add real actuarial data  
- Multiple plans (₹2L / ₹5L / ₹10L)  
- Add waiting period logic  
- Improve ML model accuracy  
- Deploy as full web app  

---

## 📌 Conclusion
This project combines machine learning and risk-based logic to simulate insurance premium calculation. While real insurers use actuarial models, this system provides a simplified and practical approximation.

---

## 📚 References
1. Principles of Risk Management and Insurance – George E. Rejda  
2. IRDAI Health Insurance Regulations  
3. https://www.kaggle.com/datasets/mirichoi0218/insurance  
4. Actuarial Mathematics for Life Contingent Risks  

---

## 📁 Project Structure├── app.py
├── model.pkl
├── googleform.csv
│
├── templates/
│ ├── home.html
│ └── output.html
│
├── static/
│ ├── css/
│ └── js/



---

## ⭐ Features
- ML-based cost prediction  
- Risk-based premium calculation  
- CSV upload support  
- Flask web interface  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
