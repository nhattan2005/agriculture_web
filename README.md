# 🌾 Crop Recommendation System using SVM from Scratch

This is a web-based machine learning application that predicts the most suitable crop to grow based on soil and climate conditions. The system uses a Support Vector Machine (SVM) classifier implemented entirely from scratch with NumPy, and is presented through a user-friendly Flask web interface.

## 📌 Project Information

- **Field**: Agriculture
- **Model**: Multiclass SVM (One-vs-Rest strategy)
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Backend**: Flask (Python)
- **Goal**: Build and evaluate a machine learning model using real-world data and deploy it as a web application.

## 📊 Dataset

- **Source**: [Crop Recommendation Dataset on Kaggle](https://www.kaggle.com/code/atharvaingle/crop-recommendation-system/input)
- **Features**:
  - `N`: Nitrogen level in the soil
  - `P`: Phosphorus level
  - `K`: Potassium level
  - `temperature`: in °C
  - `humidity`: in %
  - `ph`: soil pH value
  - `rainfall`: in mm
- **Label**: Type of crop to recommend


## 🚀 How to Run the Project

```bash
git clone https://github.com/your-username/crop-recommendation-system.git
cd crop-recommendation-system
python app.py
