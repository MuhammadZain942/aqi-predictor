import os
import joblib
import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

def main():
    print("🤖 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 1. Retrieve Feature Group and View
    aqi_fg = fs.get_feature_group(name="aqi_data_rawalpindi", version=1)
    
    try:
        feature_view = fs.get_feature_view(name="aqi_view_rawalpindi", version=1)
    except:
        print("🆕 Creating new Feature View...")
        feature_view = fs.create_feature_view(
            name="aqi_view_rawalpindi",
            version=1,
            query=aqi_fg.select_all(),
            labels=["aqi"]
        )

    # 2. Fetch Training Data
    print("🧠 Fetching training data and performing split...")
    # We use get_train_test_split if it exists, or create one
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    
    # 3. Clean Features (Remove non-numeric/metadata columns)
    drop_cols = ["date", "timestamp", "date_str", "city"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # 4. Model Selection
    models = {
        "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0)
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        print(f"📊 {name} - R2 Score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model

    # 5. Save the Winner
    print(f"🏆 Winner: {best_model.__class__.__name__}")
    joblib.dump(best_model, 'aqi_model.pkl')
    
    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="aqi_model_rawalpindi",
        metrics={"r2": best_score},
        description="AQI Predictor for Rawalpindi"
    )
    aqi_model.save('aqi_model.pkl')
    print("✅ Model registered successfully!")

if __name__ == "__main__":
    main()