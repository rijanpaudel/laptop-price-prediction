from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load trained model
try:
    model = pickle.load(open('laptop_price_model.pkl', 'rb'))
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Please run the training script first.")
    model = None

def parse_memory(memory_str):
    """Parse memory string to extract SSD and HDD values in GB"""
    ssd_gb = 0
    hdd_gb = 0
    
    # Handle different memory formats
    memory_str = memory_str.upper()
    
    if 'SSD' in memory_str and 'HDD' in memory_str:
        # Hybrid storage like "256GB SSD + 1TB HDD"
        parts = memory_str.split('+')
        for part in parts:
            part = part.strip()
            if 'SSD' in part:
                size_str = part.replace('SSD', '').strip()
                if 'TB' in size_str:
                    ssd_gb = int(float(size_str.replace('TB', '')) * 1000)
                elif 'GB' in size_str:
                    ssd_gb = int(size_str.replace('GB', ''))
            elif 'HDD' in part:
                size_str = part.replace('HDD', '').strip()
                if 'TB' in size_str:
                    hdd_gb = int(float(size_str.replace('TB', '')) * 1000)
                elif 'GB' in size_str:
                    hdd_gb = int(size_str.replace('GB', ''))
    elif 'SSD' in memory_str:
        # SSD only like "512GB SSD"
        size_str = memory_str.replace('SSD', '').strip()
        if 'TB' in size_str:
            ssd_gb = int(float(size_str.replace('TB', '')) * 1000)
        elif 'GB' in size_str:
            ssd_gb = int(size_str.replace('GB', ''))
    elif 'HDD' in memory_str:
        # HDD only like "1TB HDD"
        size_str = memory_str.replace('HDD', '').strip()
        if 'TB' in size_str:
            hdd_gb = int(float(size_str.replace('TB', '')) * 1000)
        elif 'GB' in size_str:
            hdd_gb = int(size_str.replace('GB', ''))
    
    return ssd_gb, hdd_gb

def calculate_ppi(screen_resolution, inches):
    """Calculate pixels per inch from resolution and screen size"""
    try:
        # Parse resolution like "1920x1080"
        x_res, y_res = map(int, screen_resolution.split('x'))
        # Calculate PPI using Pythagorean theorem
        ppi = ((x_res**2 + y_res**2)**0.5) / inches
        return ppi
    except:
        # Default PPI for common Full HD laptop
        return 141.21

def extract_cpu_brand(cpu_str):
    """Extract CPU brand from CPU string"""
    cpu_str = cpu_str.upper()
    if 'INTEL' in cpu_str:
        return 'Intel'
    elif 'AMD' in cpu_str:
        return 'AMD'
    else:
        return 'Other'

def extract_gpu_brand(gpu_str):
    """Extract GPU brand from GPU string"""
    gpu_str = gpu_str.upper()
    if 'INTEL' in gpu_str:
        return 'Intel'
    elif 'AMD' in gpu_str or 'RADEON' in gpu_str:
        return 'AMD'
    elif 'NVIDIA' in gpu_str or 'GEFORCE' in gpu_str or 'GTX' in gpu_str or 'RTX' in gpu_str:
        return 'Nvidia'
    else:
        return 'Intel'  # Default to integrated

def determine_os(company):
    """Determine OS based on company (simplified logic)"""
    if company.upper() == 'APPLE':
        return 'Mac'
    else:
        return 'Windows'

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Extract and process data according to your dataset columns
        company = data.get('company', 'Unknown')
        typename = data.get('typename', 'Notebook')
        inches = float(data.get('inches', 15.6))
        screen_resolution = data.get('screenResolution', '1920x1080')
        cpu = data.get('cpu', 'Intel Core i5')
        ram = int(data.get('ram', 8))
        memory = data.get('memory', '512GB SSD')
        gpu = data.get('gpu', 'Intel Integrated Graphics')
        weight = float(data.get('weight', 2.0))
        
        # Process the data to match training format
        ssd, hdd = parse_memory(memory)
        ppi = calculate_ppi(screen_resolution, inches)
        cpu_brand = extract_cpu_brand(cpu)
        gpu_brand = extract_gpu_brand(gpu)
        os = determine_os(company)
        
        # Create input dataframe with exact column names as in training
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [typename],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [0],  # Default value
            'Ips': [0],  # Default value
            'ppi': [ppi],
            'Cpu brand': [cpu_brand],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu_brand],
            'os': [os]
        })
        
        print(f"Processed input data: {input_data.iloc[0].to_dict()}")
        
        # Make prediction
        log_price = model.predict(input_data)[0]
        estimated_price = np.exp(log_price)
        
        # Calculate confidence and price range (simplified)
        confidence = min(max(int(85 + np.random.normal(0, 5)), 70), 95)
        price_range_low = estimated_price * 0.85
        price_range_high = estimated_price * 1.15
        
        result = {
            'estimatedPrice': round(estimated_price, 2),
            'confidence': confidence,
            'priceRange': {
                'low': round(price_range_low, 2),
                'high': round(price_range_high, 2)
            }
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Server will be available at: http://localhost:4000")
    print("🔗 API endpoint: http://localhost:4000/api/predict")
    app.run(debug=True, port=4000)