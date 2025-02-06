# 🌳 Tree Health Dashboard

The **Tree Health Dashboard** is a web-based application designed to analyze and predict tree health using a Random Forest machine learning model. The application provides interactive visualizations, single and batch predictions, and insights into the key factors affecting tree health.

## 🚀 Features

### ✅ **Tree Health Prediction**
- **Single Tree Prediction:** Users can input tree characteristics such as DBH, height, soil properties, and geographic location to predict its health status.
- **Batch Prediction:** Users can upload a CSV file containing multiple tree records, and the system will generate health predictions for each tree.

### 📊 **Feature Importance Visualization**
- The dashboard includes a **bar chart** displaying the most important features affecting tree health, helping researchers and ecologists understand the key contributing factors.

### 📖 **Informational Facts Page**
- Explains why **Random Forest** was chosen for the model and provides insights into the importance of various features in predicting tree health.

### 🎨 **Dark Mode Aesthetic**
- The interface is styled with a **dark mode design** featuring **purple accents**, ensuring a modern and visually appealing experience.

## 🛠️ Technologies Used

- **Frontend:** HTML, CSS (Dark Mode with Purple Accents), and JavaScript
- **Backend:** Flask (Python-based web framework)
- **Machine Learning Model:** Random Forest (Scikit-Learn)
- **Data Handling:** Pandas for processing CSV data
- **Visualization:** Matplotlib for feature importance graphs

## 📂 Project Structure
```
/tree_health_app
    /templates
        index.html       # Main dashboard UI
        facts.html       # Facts about model & feature importance
    /static
        style.css        # Styles for dark mode UI
        feature_importance.png # Feature importance chart
    app.py              # Flask backend
    random_forest_model.pkl  # Trained ML model
    label_encoder.pkl   # Label encoder for health status
    requirements.txt    # Dependencies for the project
    README.md           # Project documentation
```

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ShreyRoy/palantir_winter25.git
cd tree-health-dashboard
```

### 2️⃣ Create and Activate Virtual Environment
**Using Virtualenv:**
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate    # On Windows
```

**Using Conda:**
```bash
conda create -n tree_health_env python=3.9
conda activate tree_health_env
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python app.py
```

Access the dashboard at: **`http://127.0.0.1:5000/`**

## 📌 Usage

### 🔹 Predicting a Single Tree’s Health
1. Open the dashboard.
2. Enter tree parameters into the input form.
3. Click **Predict** to get the health status.

### 🔹 Batch Prediction via CSV Upload
1. Prepare a CSV file containing tree data.
2. Upload the file on the dashboard.
3. Download the results with predicted health statuses.

### 🔹 Learn About the Model
- Visit **`/facts`** to understand why **Random Forest** was chosen and see which features impact tree health the most.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.
