"""
Machine Learning Pipeline for Power Outage Prediction
Includes baseline model, XGBoost, SHAP explainability, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

class OutagePredictionPipeline:
    def __init__(self, data_path='outage_data.csv', random_seed=42):
        self.data_path = data_path
        self.random_seed = random_seed
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("📂 Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Feature selection
        feature_cols = [
            'electricity_load', 'rainfall', 'temperature', 'humidity',
            'equipment_age', 'maintenance_frequency', 'equipment_health',
            'past_outages', 'grid_complexity', 'voltage_stability', 'month'
        ]
        
        X = df[feature_cols].copy()
        y = df['outage'].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        self.feature_names = feature_cols
        self.X = X
        self.y = y
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, test_size=0.2):
        """Split data into train/test with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_seed,
            stratify=self.y
        )
        
        print(f"✂️  Split: Train {len(X_train)}, Test {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_baseline(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression baseline"""
        print("\n🔵 Training Baseline (Logistic Regression)...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        baseline = LogisticRegression(max_iter=1000, random_state=self.random_seed, class_weight='balanced')
        baseline.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = baseline.predict(X_test_scaled)
        y_pred_proba = baseline.predict_proba(X_test_scaled)[:, 1]
        
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        return baseline, scaler, X_test_scaled, metrics
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost...")
        
        # XGBoost with focus on recall (minimize false negatives)
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            eval_metric='logloss',
            verbosity=0
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        return xgb_model, X_test, metrics
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model and return metrics"""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        print(f"\n📊 {model_name} Metrics:")
        print(f"   Precision: {precision:.4f} (how many predicted outages actually happened)")
        print(f"   Recall:    {recall:.4f} (how many actual outages we caught) ⭐")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"   ⚠️  False Negatives: {fn} (missed high-risk zones!)")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': (tn, fp, fn, tp),
            'model_name': model_name
        }
    
    def generate_shap_values(self, model, X_test):
        """Generate SHAP values for model interpretability"""
        print("\n🔍 Generating SHAP explanations...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification (shap_values is list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        self.explainer = explainer
        self.shap_values = shap_values
        
        print("✅ SHAP values generated")
        return explainer, shap_values
    
    def get_feature_importance(self, model):
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = None
        
        if importances is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n🎯 Top Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        
        return None
    
    def save_model(self, model, scaler, filename='model.pkl'):
        """Save trained model and scaler"""
        with open(filename, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'features': self.feature_names}, f)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename='model.pkl'):
        """Load trained model and scaler"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 Model loaded from {filename}")
        return data['model'], data['scaler'], data['features']

def run_pipeline():
    """Run complete ML pipeline"""
    print("="*60)
    print("🔌 POWER OUTAGE PREDICTION PIPELINE")
    print("="*60)
    
    pipeline = OutagePredictionPipeline()
    
    # Prepare data
    X, y = pipeline.load_and_prepare_data()
    X_train, X_test, y_train, y_test = pipeline.split_data()
    
    # Train baseline
    baseline, scaler, X_test_scaled, baseline_metrics = pipeline.train_baseline(
        X_train, X_test, y_train, y_test
    )
    
    # Train XGBoost
    xgb_model, X_test_xgb, xgb_metrics = pipeline.train_xgboost(
        X_train, X_test, y_train, y_test
    )
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    importance_df = pipeline.get_feature_importance(xgb_model)
    
    # SHAP values
    print("\n" + "="*60)
    print("EXPLAINABILITY ANALYSIS")
    print("="*60)
    explainer, shap_vals = pipeline.generate_shap_values(xgb_model, X_test)
    
    # Save model
    pipeline.save_model(xgb_model, None, 'xgb_model.pkl')
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'pipeline': pipeline,
        'model': xgb_model,
        'X_test': X_test,
        'y_test': y_test,
        'metrics': xgb_metrics,
        'importance_df': importance_df,
        'explainer': explainer,
        'shap_values': shap_vals
    }

if __name__ == "__main__":
    results = run_pipeline()
