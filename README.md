# **Earthquake Severity Prediction Model**

## **Overview**
This project is a machine learning model designed to predict the severity level of an earthquake based on parameters such as magnitude, depth, and distance to the nearest city. The model is trained using real earthquake data from the USGS API and provides predictions to classify the severity into three levels: **High**, **Medium**, and **Low**.

---

## **Features**
- **Dynamic Data Fetching:** Retrieves earthquake data from the USGS API in real-time.
- **Geographical Analysis:** Calculates the distance to the nearest major city using the Geopy library.
- **Severity Classification:** Classifies earthquakes into High, Medium, or Low severity based on predefined thresholds.
- **Interactive Predictions:** Accepts user inputs for manual prediction of earthquake severity.

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8 or higher
- Pip (Python package manager)
- Geopy and required Python libraries (`pip install geopy pandas numpy scikit-learn requests`)

---

### **2. Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/Harsh-Agarwall/Earthquake_severity_prediction
   ```
### **Model Logic**
**Data Fetching:**
Earthquake data is fetched dynamically from the USGS API.
Major cities in India are used for calculating distances.

**Severity Classification:**

High Severity: Magnitude ≥ 6.5 or Magnitude ≥ 5.5 and distance ≤ 50 km.
Medium Severity: Magnitude ≥ 4.5 or Magnitude ≥ 4.0 and distance ≤ 100 km.
Low Severity: Remaining cases.

## **Machine Learning:**
Random Forest Classifier is used to train the model with fetched data.

## Future Enhancements
- Expand the city database to global locations.

- Integrate with a web application for real-time predictions.

- Store earthquake data in a database for analysis.
