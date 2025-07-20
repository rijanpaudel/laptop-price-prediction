import pandas as pd
import numpy as np
import pickle
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error

print("🔄 Loading dataset...")

# Load dataset - make sure your CSV has columns: Company,TypeName,Inches,ScreenResolution,Cpu,Ram,Memory,Gpu,Weight,Price
try:
    df = pd.read_csv('laptop_data.csv')
    print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
except FileNotFoundError:
    print("❌ laptop_data.csv not found. Please ensure the file exists in the current directory.")
    exit(1)

print("\n🔄 Preprocessing data...")

# Create a copy for processing
df_processed = df.copy()

# Clean and process RAM column
print("📝 Processing RAM...")
df_processed['Ram'] = df_processed['Ram'].astype(str).str.replace('GB', '', regex=False).str.replace('gb', '', regex=False)
df_processed['Ram'] = pd.to_numeric(df_processed['Ram'], errors='coerce')
# Fill any NaN values with median
df_processed['Ram'].fillna(df_processed['Ram'].median(), inplace=True)
df_processed['Ram'] = df_processed['Ram'].astype('int32')

# Clean and process Weight column
print("📝 Processing Weight...")
df_processed['Weight'] = df_processed['Weight'].astype(str).str.replace('kg', '', regex=False).str.replace('Kg', '', regex=False)
df_processed['Weight'] = pd.to_numeric(df_processed['Weight'], errors='coerce')
# Fill any NaN values with median
df_processed['Weight'].fillna(df_processed['Weight'].median(), inplace=True)
df_processed['Weight'] = df_processed['Weight'].astype('float32')

# Add missing columns that might not be in your dataset
print("📝 Adding missing features...")
df_processed['Touchscreen'] = 0  # Default to no touchscreen
df_processed['Ips'] = 0  # Default to no IPS

# Process Screen Resolution and calculate PPI
print("📝 Processing Screen Resolution and calculating PPI...")
def extract_resolution_and_calculate_ppi(row):
    """Extract X and Y resolution and calculate PPI"""
    try:
        resolution = str(row['ScreenResolution'])
        inches = float(row['Inches'])
        
        # Extract resolution numbers using regex
        resolution_pattern = r'(\d+)x(\d+)'
        match = re.search(resolution_pattern, resolution)
        
        if match:
            x_res = int(match.group(1))
            y_res = int(match.group(2))
        else:
            # Default to Full HD if can't parse
            x_res, y_res = 1920, 1080
            
        # Calculate PPI
        ppi = ((x_res**2 + y_res**2)**0.5) / inches
        return pd.Series([x_res, y_res, ppi])
    except:
        # Default values
        return pd.Series([1920, 1080, 141.21])

df_processed[['X_res', 'Y_res', 'ppi']] = df_processed.apply(extract_resolution_and_calculate_ppi, axis=1)

# Drop intermediate columns
df_processed.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Process CPU to extract brand
print("📝 Processing CPU...")
def extract_cpu_brand(cpu_str):
    cpu_str = str(cpu_str).upper()
    if 'INTEL' in cpu_str:
        return 'Intel'
    elif 'AMD' in cpu_str:
        return 'AMD'
    else:
        return 'Other'

df_processed['Cpu brand'] = df_processed['Cpu'].apply(extract_cpu_brand)
df_processed.drop(columns=['Cpu'], inplace=True)

# Process GPU to extract brand
print("📝 Processing GPU...")
def extract_gpu_brand(gpu_str):
    gpu_str = str(gpu_str).upper()
    if 'INTEL' in gpu_str:
        return 'Intel'
    elif 'AMD' in gpu_str or 'RADEON' in gpu_str:
        return 'AMD'
    elif 'NVIDIA' in gpu_str or 'GEFORCE' in gpu_str or 'GTX' in gpu_str or 'RTX' in gpu_str:
        return 'Nvidia'
    else:
        return 'Intel'  # Default to integrated

df_processed['Gpu brand'] = df_processed['Gpu'].apply(extract_gpu_brand)
df_processed.drop(columns=['Gpu'], inplace=True)

# Process Memory to extract SSD and HDD
print("📝 Processing Memory/Storage...")
def extract_ssd(memory_str):
    """Extract SSD capacity in GB"""
    try:
        memory_str = str(memory_str).upper()
        if 'SSD' in memory_str:
            # Extract number before SSD
            ssd_pattern = r'(\d+(?:\.\d+)?)\s*(?:TB|GB)?\s*SSD'
            matches = re.findall(ssd_pattern, memory_str)
            if matches:
                value = float(matches[-1])  # Take last match
                # Check if it's TB
                if 'TB' in memory_str and 'SSD' in memory_str:
                    tb_pattern = r'(\d+(?:\.\d+)?)\s*TB\s*SSD'
                    if re.search(tb_pattern, memory_str):
                        return int(value * 1000)
                return int(value)
        return 0
    except:
        return 0

def extract_hdd(memory_str):
    """Extract HDD capacity in GB"""
    try:
        memory_str = str(memory_str).upper()
        if 'HDD' in memory_str:
            # Extract number before HDD
            hdd_pattern = r'(\d+(?:\.\d+)?)\s*(?:TB|GB)?\s*HDD'
            matches = re.findall(hdd_pattern, memory_str)
            if matches:
                value = float(matches[-1])  # Take last match
                # Check if it's TB
                if 'TB' in memory_str and 'HDD' in memory_str:
                    tb_pattern = r'(\d+(?:\.\d+)?)\s*TB\s*HDD'
                    if re.search(tb_pattern, memory_str):
                        return int(value * 1000)
                return int(value)
        return 0
    except:
        return 0

df_processed['SSD'] = df_processed['Memory'].apply(extract_ssd)
df_processed['HDD'] = df_processed['Memory'].apply(extract_hdd)

# If both SSD and HDD are 0, assume it's an SSD (common case)
mask = (df_processed['SSD'] == 0) & (df_processed['HDD'] == 0)
if mask.sum() > 0:
    print(f"⚠️  Found {mask.sum()} rows with no storage detected. Setting default to 512GB SSD...")
    df_processed.loc[mask, 'SSD'] = 512

df_processed.drop(columns=['Memory'], inplace=True)

# Add OS column based on company
print("📝 Adding OS information...")
def determine_os(company):
    if str(company).upper() == 'APPLE':
        return 'Mac'
    else:
        return 'Windows'

df_processed['os'] = df_processed['Company'].apply(determine_os)

# Clean up company names
print("📝 Cleaning company names...")
df_processed['Company'] = df_processed['Company'].str.strip().str.title()

# Clean up type names
print("📝 Cleaning type names...")
df_processed['TypeName'] = df_processed['TypeName'].str.strip().str.title()

# Remove any rows with missing price
print("📝 Cleaning price data...")
df_processed = df_processed.dropna(subset=['Price'])
df_processed = df_processed[df_processed['Price'] > 0]

print(f"📊 Final processed dataset shape: {df_processed.shape}")
print(f"📋 Final columns: {list(df_processed.columns)}")

# Display some statistics
print("\n📈 Dataset Statistics:")
print(f"Price range: ${df_processed['Price'].min():.2f} - ${df_processed['Price'].max():.2f}")
print(f"Average price: ${df_processed['Price'].mean():.2f}")
print(f"Companies: {df_processed['Company'].nunique()} unique")
print(f"CPU brands: {df_processed['Cpu brand'].unique()}")
print(f"GPU brands: {df_processed['Gpu brand'].unique()}")

# Prepare features and target
print("\n🎯 Preparing features and target...")
X = df_processed.drop(columns=['Price'])
y = np.log(df_processed['Price'])  # Apply log transformation to target

# Define categorical and numerical features
categorical_features = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
numerical_features = [col for col in X.columns if col not in categorical_features]

print(f"📝 Categorical features: {categorical_features}")
print(f"📝 Numerical features: {numerical_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set size: {X_train.shape[0]}")
print(f"📊 Test set size: {X_test.shape[0]}")

# Train the model
print("\n🏋️  Training model...")
pipeline.fit(X_train, y_train)

# Evaluate the model
print("\n📏 Evaluating model...")
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Convert back from log space for MAE calculation
train_mae = mean_absolute_error(np.exp(y_train), np.exp(y_pred_train))
test_mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))

print(f"📈 Training R² Score: {train_r2:.4f}")
print(f"📈 Test R² Score: {test_r2:.4f}")
print(f"📈 Training MAE: ${train_mae:.2f}")
print(f"📈 Test MAE: ${test_mae:.2f}")

# Save the model
print("\n💾 Saving model...")
with open('laptop_price_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model training complete and saved to 'laptop_price_model.pkl'")

# Test the model with a sample prediction
print("\n🧪 Testing model with sample prediction...")
sample_data = pd.DataFrame({
    'Company': ['HP'],
    'TypeName': ['Notebook'],
    'Ram': [8],
    'Weight': [2.1],
    'Touchscreen': [0],
    'Ips': [0],
    'ppi': [141.21],
    'Cpu brand': ['Intel'],
    'HDD': [0],
    'SSD': [512],
    'Gpu brand': ['Intel'],
    'os': ['Windows']
})

try:
    sample_prediction = pipeline.predict(sample_data)[0]
    sample_price = np.exp(sample_prediction)
    print(f"🎯 Sample prediction: ${sample_price:.2f}")
except Exception as e:
    print(f"❌ Sample prediction failed: {e}")

print("\n🎉 Training script completed successfully!")
print("🚀 You can now run the Flask server with: python app.py")