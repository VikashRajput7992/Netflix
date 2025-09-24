#!/usr/bin/env python3
"""
Netflix Recommendation System - Main Training Script

This script orchestrates the complete training pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Model saving and reporting

Usage: python src/train_models.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_utils import DataProcessor, FeatureEngineering
from src.model_utils import ModelTrainer, ModelEvaluator

def main():
    """Main training pipeline"""
    
    print("🎬 NETFLIX RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize components
    data_processor = DataProcessor(data_dir=project_root / "data")
    feature_engineer = FeatureEngineering()
    model_trainer = ModelTrainer(models_dir=project_root / "models" / "trained", 
                                results_dir=project_root / "results" / "metrics")
    model_evaluator = ModelEvaluator(results_dir=project_root / "results" / "reports")
    
    # Step 1: Load and preprocess data
    print("\n📁 STEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 40)
    
    try:
        # Load raw data
        df_movies_raw = data_processor.load_raw_data()
        print(f"✅ Loaded raw movies data: {len(df_movies_raw)} records")
        
        # Preprocess movies data
        df_movies = data_processor.preprocess_movies_data(df_movies_raw)
        print(f"✅ Preprocessed movies data: {df_movies.shape}")
        
        # Create synthetic users
        df_users = data_processor.create_synthetic_users(n_users=5000)
        print(f"✅ Created synthetic users: {len(df_users)} profiles")
        
        # Create interactions
        df_interactions = data_processor.create_user_movie_interactions(
            df_users, df_movies, n_interactions=50000
        )
        print(f"✅ Created synthetic interactions: {len(df_interactions)} pairs")
        
        # Save processed data
        data_processor.save_processed_data(df_movies, df_users, df_interactions)
        
    except Exception as e:
        print(f"❌ Error in data processing: {str(e)}")
        return
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    
    try:
        # Engineer features
        df_features = feature_engineer.engineer_features(df_interactions, df_users, df_movies)
        print(f"✅ Feature engineering completed: {df_features.shape}")
        
        # Prepare features for ML
        X, y, feature_columns = feature_engineer.prepare_features(df_features)
        print(f"✅ Features prepared: {len(feature_columns)} features")
        
        # Save feature data
        final_data_dir = project_root / "data" / "final"
        final_data_dir.mkdir(exist_ok=True)
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Save splits
        X_train.to_csv(final_data_dir / "X_train.csv", index=False)
        X_val.to_csv(final_data_dir / "X_val.csv", index=False)
        X_test.to_csv(final_data_dir / "X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(final_data_dir / "y_train.csv", index=False)
        pd.DataFrame(y_val).to_csv(final_data_dir / "y_val.csv", index=False)
        pd.DataFrame(y_test).to_csv(final_data_dir / "y_test.csv", index=False)
        
        print("✅ Train/val/test splits saved")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return
    
    # Step 3: Feature Selection and Scaling
    print("\n🎯 STEP 3: FEATURE SELECTION AND SCALING")
    print("-" * 40)
    
    try:
        # Feature selection
        X_train_selected, selected_features, selector = model_trainer.feature_selection(
            X_train, y_train, method='statistical', k=30
        )
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        print(f"✅ Feature selection completed: {len(selected_features)} features selected")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        print("✅ Feature scaling completed")
        
    except Exception as e:
        print(f"❌ Error in feature selection/scaling: {str(e)}")
        return
    
    # Step 4: Model Training
    print("\n🤖 STEP 4: MODEL TRAINING")
    print("-" * 40)
    
    try:
        # Train baseline models
        trained_models, baseline_results = model_trainer.train_baseline_models(
            X_train_scaled, X_val_scaled, y_train, y_val
        )
        
        # Find best model
        best_model_name = max(baseline_results.keys(), key=lambda k: baseline_results[k]['val_f1'])
        print(f"✅ Best baseline model: {best_model_name}")
        
        # Hyperparameter tuning for best model
        print(f"\n🔧 Hyperparameter tuning for {best_model_name}...")
        tuned_model, best_params, best_score = model_trainer.hyperparameter_tuning(
            np.vstack([X_train_scaled, X_val_scaled]),
            np.hstack([y_train, y_val]),
            model_name=best_model_name
        )
        
        print(f"✅ Hyperparameter tuning completed: CV Score = {best_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return
    
    # Step 5: Final Evaluation
    print("\n📊 STEP 5: FINAL EVALUATION")
    print("-" * 40)
    
    try:
        # Evaluate on test set
        final_metrics = model_trainer.evaluate_final_model(
            tuned_model, X_test_scaled, y_test, best_model_name
        )
        
        # Generate comparison report
        comparison_df = model_evaluator.compare_models(baseline_results)
        print(f"✅ Model comparison completed")
        
        # Save feature importance
        feature_importance_df = model_trainer.save_feature_importance(
            tuned_model, selected_features, best_model_name
        )
        
    except Exception as e:
        print(f"❌ Error in final evaluation: {str(e)}")
        return
    
    # Step 6: Model Saving
    print("\n💾 STEP 6: MODEL SAVING")
    print("-" * 40)
    
    try:
        # Save final model
        saved_files = model_trainer.save_model(
            tuned_model, scaler, selected_features, best_model_name, final_metrics
        )
        
        # Generate summary report
        summary = model_evaluator.generate_summary_report(
            best_model_name, final_metrics, feature_importance_df
        )
        
        print("✅ All artifacts saved successfully!")
        
    except Exception as e:
        print(f"❌ Error in model saving: {str(e)}")
        return
    
    # Final Summary
    print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✅ Best Model: {best_model_name}")
    print(f"✅ Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"✅ Test F1-Score: {final_metrics['f1']:.4f}")
    print(f"✅ Features Used: {len(selected_features)}")
    print(f"✅ Model File: {Path(saved_files['model_file']).name}")
    print("\n🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()