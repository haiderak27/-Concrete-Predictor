import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🏗️ CONCRETE STRENGTH PREDICTOR - TRAINING")
print("=" * 60)

# STEP 1: Load the data with TAB separator
print("\n📊 Loading data...")
try:
    # IMPORTANT: Use tab separator ('\t') because your file uses tabs
    df = pd.read_csv('../data/concrete_data.csv', sep='\t')
    print(f"   ✅ Found {len(df)} concrete mixes")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("   Trying with comma separator...")
    try:
        df = pd.read_csv('../data/concrete_data.csv')
        print(f"   ✅ Found {len(df)} concrete mixes with comma separator")
    except Exception as e2:
        print(f"   ❌ ERROR: {e2}")
        exit(1)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Show what we found
print("\n📋 Column names in your CSV:")
for i, col in enumerate(df.columns):
    print(f"   {i+1}. '{col}'")

# Define the correct feature columns
feature_cols = [
    'cement', 
    'blast_furnace_slag', 
    'fly_ash', 
    'water', 
    'superplasticizer', 
    'coarse_aggregate', 
    'fine_aggregate', 
    'age'
]

target_col = 'concrete_compressive_strength'

# Check if all columns exist
missing_cols = []
for col in feature_cols + [target_col]:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"\n❌ ERROR: Missing columns: {missing_cols}")
    print("\n💡 The column names might have extra characters.")
    print("   Looking for similar column names...")
    for col in missing_cols:
        for df_col in df.columns:
            if col.replace('_', '') in df_col.replace('_', '').replace(' ', ''):
                print(f"   Did you mean '{df_col}' instead of '{col}'?")
    exit(1)

print(f"\n✅ All columns found successfully!")

# Prepare features and target
X = df[feature_cols]
y = df[target_col]

print(f"   Strength range: {y.min():.2f} to {y.max():.2f} MPa")

# STEP 2: Split data
print("\n🔧 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# STEP 3: Scale features
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 4: Train model
print("\n🌳 Training Random Forest model...")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
print("   ✅ Model training complete!")

# STEP 5: Evaluate model
print("\n📊 Evaluating model performance...")

y_pred = rf.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"   R² Score: {r2:.4f} (closer to 1 is better)")
print(f"   MAE: {mae:.2f} MPa (average error)")
print(f"   RMSE: {rmse:.2f} MPa (typical error)")

# STEP 6: Feature importance
print("\n🔍 Most important features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.2%}")

# STEP 7: Save artifacts
print("\n💾 Saving artifacts...")

import os
os.makedirs('../models', exist_ok=True)

try:
    joblib.dump(rf, '../models/concrete_strength_model.pkl')
    print("   ✅ Model saved to models/concrete_strength_model.pkl")
    
    joblib.dump(scaler, '../models/concrete_scaler.pkl')
    print("   ✅ Scaler saved to models/concrete_scaler.pkl")
    
    metadata = {
        'feature_cols': feature_cols,
        'target_column': target_col,
        'r2_score': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'feature_importance': feature_importance.to_dict('records')
    }
    joblib.dump(metadata, '../models/model_metadata.pkl')
    print("   ✅ Metadata saved to models/model_metadata.pkl")
    
except Exception as e:
    print(f"   ❌ ERROR saving artifacts: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE! You can now use the model.")
print("=" * 60)

# Show sample predictions
print("\n🎯 Sample predictions on test data:")
sample_indices = [0, 1, 2]
for i in sample_indices:
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"   Sample {i+1}: Actual: {actual:.2f} MPa, Predicted: {predicted:.2f} MPa, Error: {error:.2f} MPa")