import os
import joblib
import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

def main():
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. Retrieve Feature View
    try:
        aqi_fg = fs.get_feature_group(name="aqi_data_rawalpindi", version=1)
        feature_view = fs.get_feature_view(name="aqi_view_rawalpindi", version=1)
    except:
        # Create if missing
        aqi_fg = fs.get_feature_group(name="aqi_data_rawalpindi", version=1)
        feature_view = fs.create_feature_view(
            name="aqi_view_rawalpindi",
            version=1,
            query=aqi_fg.select_all(),
            labels=["aqi"]
        )

    # 3. Create Training Data
    print("🧠 Fetching training data...")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    
    # Drop non-feature columns
    drop_cols = ["date", "timestamp", "date_str", "city"]
    actual_drop_cols = [c for c in drop_cols if c in X_train.columns]
    X_train = X_train.drop(columns=actual_drop_cols)
    X_test = X_test.drop(columns=actual_drop_cols)

    # 4. Define the Contenders (3 Models)
    models = {
        "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Ridge_Regression": Ridge(alpha=1.0)
    }

    best_name = None
    best_score = -999
    best_model = None
    best_metrics = {}

    print(f"🥊 Starting Battle of Models (Training on {len(X_train)} rows)...")
    print("-" * 50)

    # 5. Train & Evaluate Loop
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   👉 {name}: R2 = {r2*100:.2f}% | MAE = {mae:.2f}")
        
        # Keep the winner
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model
            best_metrics = {"r2": r2, "mae": mae}

    print("-" * 50)
    print(f"🏆 WINNER: {best_name} with R2 Score of {best_score*100:.2f}%")

    # 6. Register ONLY the Best Model
    print("💾 Saving the Champion to Registry...")
    mr = project.get_model_registry()
    
    # Save to local file
    joblib.dump(best_model, 'aqi_model.pkl')
    
    # Create model card in Hopsworks
    # Note: We increment version automatically if it exists
    aqi_model = mr.python.create_model(
        name="aqi_model_rawalpindi",
        metrics=best_metrics,
        description=f"Best Model: {best_name}. Predicts AQI based on Weather."
    )
    aqi_model.save('aqi_model.pkl')
    
    print("🎉 Success! The best model is saved in the cloud.")

if __name__ == "__main__":
    main()