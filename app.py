from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)

# Load the Random Forest model
rf_model = load('model.pkl')
le = load('label_encoder.pkl')
features = ['Latitude', 'Longitude', 'DBH', 'Tree_Height', 'Crown_Width_North_South', 'Crown_Width_East_West', 'Slope', 'Elevation', 'Temperature', 'Humidity',
            'Soil_TN', 'Soil_TP', 'Soil_AP', 'Soil_AN', 'Menhinick_Index', 'Gleason_Index', 
            'Disturbance_Level', 'Fire_Risk_Index']

def generate_plot_url(rf_model, features):
    importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    
    # Set bar colors to the purple accent
    bar_color = '#bb86fc'
    plt.barh(features, importances, color=bar_color, align='center')
    
    plt.xlabel('Importance', color='white')
    plt.ylabel('Feature', color='white')
    plt.title('Feature Importance', color='white')
    
    # Adjust the background to match the dark mode aesthetic
    plt.gca().set_facecolor('#121212')  # Dark gray background for plot
    plt.gcf().set_facecolor('#121212')  # Dark gray background for figure
    
    # Change axis colors to light gray for better readability
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    
    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def save_feature_importance_chart(rf_model, features):
    importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    
    # Set bar colors to the purple accent
    bar_color = '#bb86fc'
    plt.barh(features, importances, color=bar_color, align='center')
    
    plt.xlabel('Importance', color='white')
    plt.ylabel('Feature', color='white')
    plt.title('Feature Importance', color='white')
    
    # Adjust the background to match the dark mode aesthetic
    plt.gca().set_facecolor('#121212')  # Dark gray background for plot
    plt.gcf().set_facecolor('#121212')  # Dark gray background for figure
    
    # Change axis colors to light gray for better readability
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')

    # Save the plot to the static folder
    plt.savefig('static/feature_importance.png')
    plt.close()

# Call this function when starting the Flask app
save_feature_importance_chart(rf_model, features)

# Home route for feature importance visualization
@app.route('/')
def index():
    # Generate feature importance plot
    plot_url = generate_plot_url(rf_model, features)
    return render_template('index.html', plot_url=plot_url, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = request.form.to_dict(flat=True)
        input_df = pd.DataFrame([input_data], columns=features)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
        
        # Make a prediction
        prediction = rf_model.predict(input_df)
        health_status = le.inverse_transform(prediction)[0]  # Translate number to health status
        
        # Generate feature importance plot
        plot_url = generate_plot_url(rf_model, features)
        
        # Render the result along with the form
        return render_template('index.html', prediction=health_status, form_data=input_data, plot_url=plot_url)
    except Exception as e:
        # Pass form_data and plot_url even in case of an error
        input_data = request.form.to_dict(flat=True)  # Ensure form data is preserved
        plot_url = generate_plot_url(rf_model, features)
        return render_template('index.html', error=str(e), form_data=input_data, plot_url=plot_url)


# Route for file upload and batch predictions
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Read the uploaded CSV file
        if file:
            data = pd.read_csv(file)

            # Check that required features exist in the uploaded file
            missing_features = set(features) - set(data.columns)
            if missing_features:
                return jsonify({'error': f'Missing required columns: {missing_features}'})

            # Make predictions
            data['Predicted_Health_Status'] = le.inverse_transform(rf_model.predict(data[features]))

            # Select only Plot_ID, Health_Status, and Predicted_Health_Status for the output
            if 'Plot_ID' not in data.columns or 'Health_Status' not in data.columns:
                return jsonify({'error': 'The file must contain Plot_ID and Health_Status columns.'})
            
            output_data = data[['Plot_ID', 'Health_Status', 'Predicted_Health_Status']]

            # Save the results to a CSV file
            output_filename = 'batch_predictions.csv'
            output_data.to_csv(output_filename, index=False)

            return jsonify({'message': 'Predictions completed successfully!', 'file': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/facts')
def facts():
    return render_template('facts.html')

if __name__ == '__main__':
    app.run(debug=True)
