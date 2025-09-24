"""
Netflix Recommendation System - Model Training Utilities

This module contains utilities for training, evaluating, and saving
machine learning models for the Netflix recommendation system.
"""

import joblib
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb

class ModelTrainer:
    """Model training and evaluation utilities"""
    
    def __init__(self, models_dir="models/trained", results_dir="results/metrics"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500)
        }
        
        self.trained_models = {}
        self.results = {}
    
    def feature_selection(self, X_train, y_train, method='statistical', k=30):
        """Perform feature selection"""
        
        if method == 'statistical':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            
            return X_selected, selected_features, selector
        
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Get feature importance and select top k
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(k)['feature'].tolist()
            X_selected = X_train[selected_features].values
            
            return X_selected, selected_features, rf
        
        else:
            raise ValueError("Method must be 'statistical' or 'random_forest'")
    
    def train_baseline_models(self, X_train, X_val, y_train, y_val):
        """Train baseline models"""
        
        print("Training baseline models...")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Get probabilities for AUC
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_val, y_train_pred, y_val_pred, 
                                            y_train_proba, y_val_proba)
            
            # Store results
            self.trained_models[model_name] = model
            self.results[model_name] = metrics
            
            print(f"  Validation F1: {metrics['val_f1']:.4f}")
            print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
        
        return self.trained_models, self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost', cv=3):
        """Perform hyperparameter tuning for specified model"""
        
        print(f"\nTuning hyperparameters for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
            
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'Neural Network':
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
            base_model = MLPClassifier(random_state=42, max_iter=500)
            
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Perform randomized search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        print(f"  Best CV Score: {search.best_score_:.4f}")
        print(f"  Best Parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def evaluate_final_model(self, model, X_test, y_test, model_name):
        """Evaluate final model on test set"""
        
        print(f"\nEvaluating {model_name} on test set...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test F1-Score: {metrics['f1']:.4f}")
        print(f"  Test Precision: {metrics['precision']:.4f}")
        print(f"  Test Recall: {metrics['recall']:.4f}")
        print(f"  Test ROC-AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def save_model(self, model, scaler, feature_columns, model_name, performance_metrics):
        """Save trained model with all artifacts"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model artifacts
        artifacts = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'model_name': model_name,
            'model_performance': performance_metrics,
            'training_date': timestamp,
            'model_version': '1.0.0'
        }
        
        # Save with joblib (recommended for scikit-learn)
        model_filename = self.models_dir / f"netflix_recommendation_model_{timestamp}.joblib"
        joblib.dump(artifacts, model_filename)
        
        # Save with pickle as backup
        pickle_filename = self.models_dir / f"netflix_recommendation_model_{timestamp}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Save metadata
        metadata = {
            'model_info': {
                'model_type': model_name,
                'n_features': len(feature_columns),
                'training_date': timestamp,
                'model_version': '1.0.0'
            },
            'performance_metrics': performance_metrics,
            'files': {
                'model_joblib': model_filename.name,
                'model_pickle': pickle_filename.name
            }
        }
        
        metadata_filename = self.results_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Model saved successfully!")
        print(f"   📁 Joblib: {model_filename.name}")
        print(f"   📁 Pickle: {pickle_filename.name}")
        print(f"   📁 Metadata: {metadata_filename.name}")
        
        return {
            'model_file': str(model_filename),
            'pickle_file': str(pickle_filename),
            'metadata_file': str(metadata_filename),
            'timestamp': timestamp
        }
    
    def save_feature_importance(self, model, feature_columns, model_name):
        """Save feature importance analysis"""
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"feature_importance_{timestamp}.csv"
            importance_df.to_csv(filename)
            
            print(f"✅ Feature importance saved: {filename.name}")
            return importance_df
        
        return None
    
    def _calculate_metrics(self, y_train, y_val, y_train_pred, y_val_pred, y_train_proba, y_val_proba):
        """Calculate comprehensive evaluation metrics"""
        
        return {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'val_auc': roc_auc_score(y_val, y_val_proba)
        }

class ModelEvaluator:
    """Model evaluation and comparison utilities"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def compare_models(self, results_dict):
        """Compare multiple models and create comparison report"""
        
        # Convert results to DataFrame
        comparison_df = pd.DataFrame(results_dict).T
        comparison_df = comparison_df.round(4)
        
        # Sort by validation F1 score
        comparison_df = comparison_df.sort_values('val_f1', ascending=False)
        
        # Save comparison
        filename = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(filename)
        
        print(f"✅ Model comparison saved: {filename}")
        
        return comparison_df
    
    def generate_summary_report(self, best_model_name, best_performance, feature_importance_df=None):
        """Generate final project summary report"""
        
        summary = {
            'project_info': {
                'name': 'Netflix Recommendation System',
                'completion_date': datetime.now().isoformat(),
                'status': 'Completed'
            },
            'best_model': {
                'name': best_model_name,
                'performance': best_performance
            },
            'feature_analysis': {
                'total_features': len(feature_importance_df) if feature_importance_df is not None else 0,
                'top_features': feature_importance_df.head(10).to_dict() if feature_importance_df is not None else {}
            }
        }
        
        # Save summary
        filename = self.results_dir / "project_summary.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Project summary saved: {filename}")
        
        return summary