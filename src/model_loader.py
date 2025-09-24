#!/usr/bin/env python3
"""
Netflix Recommendation System - Model Loader and Display

This script loads trained models from the models folder and displays:
- Model information and metadata
- Performance metrics
- Feature importance
- Sample predictions

Usage: python src/model_loader.py
"""

import os
import sys
import json
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class NetflixModelLoader:
    """Load and display Netflix recommendation models"""
    
    def __init__(self, models_dir="models/trained", results_dir="results"):
        self.models_dir = Path(project_root) / models_dir
        self.results_dir = Path(project_root) / results_dir
        self.model = None
        self.model_metadata = None
        
    def list_available_models(self):
        """List all available model files"""
        print("🔍 AVAILABLE MODELS")
        print("=" * 50)
        
        model_files = []
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.joblib"):
                if "netflix_recommendation_model" in file.name:
                    model_files.append(file)
                    
        if not model_files:
            print("❌ No models found!")
            return []
            
        for i, model_file in enumerate(model_files, 1):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"{i}. {model_file.name}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
        return model_files
    
    def load_model(self, model_path=None):
        """Load a model from file"""
        if model_path is None:
            # Load the latest model
            model_files = list(self.models_dir.glob("*netflix_recommendation_model*.joblib"))
            if not model_files:
                print("❌ No models found!")
                return False
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            print(f"📁 Loading model: {model_path.name}")
            
            # Load model artifacts
            artifacts = joblib.load(model_path)
            
            self.model = artifacts.get('model')
            self.scaler = artifacts.get('scaler')
            self.feature_columns = artifacts.get('feature_columns', [])
            self.model_metadata = {
                'model_name': artifacts.get('model_name', 'Unknown'),
                'model_version': artifacts.get('model_version', '1.0.0'),
                'training_date': artifacts.get('training_date', 'Unknown'),
                'model_performance': artifacts.get('model_performance', {}),
                'n_features': len(self.feature_columns)
            }
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def display_model_info(self):
        """Display comprehensive model information"""
        if not self.model:
            print("❌ No model loaded!")
            return
            
        print("\n🤖 MODEL INFORMATION")
        print("=" * 50)
        
        # Basic info
        print(f"Model Type: {self.model_metadata['model_name']}")
        print(f"Version: {self.model_metadata['model_version']}")
        print(f"Training Date: {self.model_metadata['training_date']}")
        print(f"Number of Features: {self.model_metadata['n_features']}")
        
        # Model specifics
        print(f"\nModel Class: {type(self.model).__name__}")
        if hasattr(self.model, 'n_estimators'):
            print(f"N Estimators: {self.model.n_estimators}")
        if hasattr(self.model, 'max_depth'):
            print(f"Max Depth: {self.model.max_depth}")
        if hasattr(self.model, 'learning_rate'):
            print(f"Learning Rate: {self.model.learning_rate}")
    
    def display_performance_metrics(self):
        """Display model performance metrics"""
        if not self.model_metadata or 'model_performance' not in self.model_metadata:
            print("❌ No performance metrics available!")
            return
            
        print("\n📊 PERFORMANCE METRICS")
        print("=" * 50)
        
        metrics = self.model_metadata['model_performance']
        
        print(f"🎯 Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"🎯 Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"🎯 Recall:    {metrics.get('recall', 'N/A'):.4f}")
        print(f"🎯 F1-Score:  {metrics.get('f1', 'N/A'):.4f}")
        print(f"🎯 ROC-AUC:   {metrics.get('auc', 'N/A'):.4f}")
        
        # Performance interpretation
        f1_score = metrics.get('f1', 0)
        if f1_score > 0.95:
            performance_level = "🔥 EXCELLENT"
        elif f1_score > 0.85:
            performance_level = "✅ VERY GOOD"
        elif f1_score > 0.75:
            performance_level = "👍 GOOD"
        else:
            performance_level = "⚠️  NEEDS IMPROVEMENT"
            
        print(f"\nOverall Performance: {performance_level}")
    
    def display_feature_importance(self, top_n=10):
        """Display feature importance if available"""
        print(f"\n🔍 TOP {top_n} FEATURE IMPORTANCE")
        print("=" * 50)
        
        # Try to load feature importance from results
        feature_imp_files = list(self.results_dir.glob("**/feature_importance*.csv"))
        
        if feature_imp_files and hasattr(self.model, 'feature_importances_'):
            try:
                # Load feature importance CSV
                latest_file = max(feature_imp_files, key=lambda x: x.stat().st_mtime)
                feature_imp_df = pd.read_csv(latest_file, index_col=0)
                
                if 'average' in feature_imp_df.columns:
                    top_features = feature_imp_df.head(top_n)
                    
                    print(f"{'Rank':>4} {'Feature':>25} {'Importance':>12}")
                    print("-" * 45)
                    
                    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
                        importance = row['average']
                        print(f"{i:>4} {feature:>25} {importance:>11.4f}")
                        
                else:
                    print("📊 Feature importance data format not recognized")
                    
            except Exception as e:
                print(f"❌ Error loading feature importance: {str(e)}")
                
        elif hasattr(self.model, 'feature_importances_'):
            # Use model's built-in feature importance
            if self.feature_columns:
                feature_imp = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"{'Rank':>4} {'Feature':>25} {'Importance':>12}")
                print("-" * 45)
                
                for i, (_, row) in enumerate(feature_imp.head(top_n).iterrows(), 1):
                    print(f"{i:>4} {row['feature']:>25} {row['importance']:>11.4f}")
            else:
                print("📊 Feature names not available")
        else:
            print("📊 Feature importance not available for this model type")
    
    def make_sample_predictions(self, n_samples=5):
        """Make sample predictions to demonstrate model capability"""
        print(f"\n🎯 SAMPLE PREDICTIONS ({n_samples} examples)")
        print("=" * 60)
        
        # Try to load test data
        test_data_files = list(Path(project_root).glob("**/X_test.csv"))
        
        if not test_data_files:
            print("❌ No test data found for sample predictions")
            return
            
        try:
            # Load test data
            X_test = pd.read_csv(test_data_files[0])
            
            # Load corresponding y_test if available
            y_test_files = list(Path(project_root).glob("**/y_test.csv"))
            y_test = None
            if y_test_files:
                y_test = pd.read_csv(y_test_files[0])
                if 'liked' in y_test.columns:
                    y_test = y_test['liked']
                else:
                    y_test = y_test.iloc[:, 0]  # First column
            
            # Select random samples
            sample_indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
            X_sample = X_test.iloc[sample_indices]
            
            # Select relevant features if needed
            if self.feature_columns and all(col in X_sample.columns for col in self.feature_columns):
                X_sample = X_sample[self.feature_columns]
            
            # Make predictions
            predictions = self.model.predict(X_sample)
            probabilities = self.model.predict_proba(X_sample)
            
            print(f"{'Sample':>6} {'Actual':>8} {'Predicted':>11} {'Like Prob':>10} {'Confidence':>11}")
            print("-" * 50)
            
            for i, idx in enumerate(sample_indices):
                actual = y_test.iloc[idx] if y_test is not None else "N/A"
                predicted = "LIKE" if predictions[i] else "DISLIKE"
                like_prob = probabilities[i][1]
                confidence = max(probabilities[i])
                
                actual_str = "LIKE" if actual == 1 else ("DISLIKE" if actual == 0 else "N/A")
                
                print(f"{idx:>6} {actual_str:>8} {predicted:>11} {like_prob:>9.3f} {confidence:>11.3f}")
                
        except Exception as e:
            print(f"❌ Error making sample predictions: {str(e)}")
    
    def load_and_display_all(self, model_path=None):
        """Load model and display all information"""
        print("🎬 NETFLIX RECOMMENDATION MODEL ANALYZER")
        print("=" * 60)
        
        # Load model
        if not self.load_model(model_path):
            return
            
        # Display all information
        self.display_model_info()
        self.display_performance_metrics()
        self.display_feature_importance(top_n=15)
        self.make_sample_predictions(n_samples=8)
        
        print("\n" + "=" * 60)
        print("✅ Model analysis completed!")
        print("🚀 Ready for production deployment!")

def main():
    """Main function"""
    loader = NetflixModelLoader()
    
    # List available models
    models = loader.list_available_models()
    
    if not models:
        print("❌ No models found! Please train a model first.")
        return
    
    # Load and analyze the latest model
    print("📊 Analyzing the latest model...")
    loader.load_and_display_all()

if __name__ == "__main__":
    main()