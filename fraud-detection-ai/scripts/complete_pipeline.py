"""
Complete Fraud Detection Pipeline with Real Datasets
Downloads real datasets found online and trains models on them
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Complete pipeline: Download real datasets + Train models"""
    logger.info("🚀 SYNAPSE AI FRAUD DETECTION - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info("This pipeline will:")
    logger.info("1. Download REAL fraud detection datasets from online sources")
    logger.info("2. Create synthetic datasets for additional training data")
    logger.info("3. Train advanced ML models on ALL datasets combined")
    logger.info("4. Evaluate performance and save production-ready models")
    logger.info("=" * 80)
    
    # Step 1: Download Real Datasets
    logger.info("📥 STEP 1: Downloading Real Fraud Detection Datasets")
    logger.info("-" * 50)
    
    try:
        from download_real_datasets import RealDatasetDownloader
        
        downloader = RealDatasetDownloader()
        real_datasets = downloader.download_all_datasets()
        
        if real_datasets:
            logger.info(f"✅ Successfully downloaded {len(real_datasets)} real datasets!")
            
            # Print dataset details
            total_real_transactions = 0
            for name, df in real_datasets.items():
                fraud_col = df.columns[-1]
                fraud_count = df[fraud_col].sum()
                fraud_rate = fraud_count / len(df) * 100
                total_real_transactions += len(df)
                
                logger.info(f"  {name}: {len(df):,} transactions ({fraud_rate:.2f}% fraud)")
            
            logger.info(f"  TOTAL REAL DATA: {total_real_transactions:,} transactions")
        else:
            logger.warning("⚠️  No real datasets downloaded, will use synthetic data only")
            
    except Exception as e:
        logger.error(f"Failed to download real datasets: {e}")
        logger.info("Continuing with synthetic datasets only...")
    
    # Step 2: Create Synthetic Datasets
    logger.info("\n🔬 STEP 2: Creating Synthetic Datasets")
    logger.info("-" * 50)
    
    try:
        from data_acquisition import FraudDatasetManager
        
        manager = FraudDatasetManager()
        synthetic_datasets = {
            'synthetic_general': manager.create_synthetic_dataset(100000),
            'synthetic_paysim': manager.create_paysim_style_dataset(50000),
            'synthetic_ecommerce': manager.create_e_commerce_dataset(75000)
        }
        
        # Save synthetic datasets
        manager.save_datasets(synthetic_datasets)
        
        total_synthetic = sum(len(df) for df in synthetic_datasets.values())
        logger.info(f"✅ Created {len(synthetic_datasets)} synthetic datasets")
        logger.info(f"  TOTAL SYNTHETIC DATA: {total_synthetic:,} transactions")
        
    except Exception as e:
        logger.error(f"Failed to create synthetic datasets: {e}")
        return False
    
    # Step 3: Train Models on Combined Data
    logger.info("\n🤖 STEP 3: Training Advanced ML Models")
    logger.info("-" * 50)
    
    try:
        from train_multi_dataset import MultiDatasetFraudTrainer
        
        trainer = MultiDatasetFraudTrainer()
        
        # Load all datasets (real + synthetic)
        all_datasets = trainer.load_datasets()
        
        if not all_datasets:
            logger.error("No datasets available for training!")
            return False
        
        logger.info(f"📊 Training on {len(all_datasets)} datasets:")
        total_training_data = 0
        for name, df in all_datasets.items():
            logger.info(f"  {name}: {len(df):,} transactions")
            total_training_data += len(df)
        
        logger.info(f"  TOTAL TRAINING DATA: {total_training_data:,} transactions")
        
        # Prepare features from all datasets
        X, y = trainer.prepare_features(all_datasets)
        
        logger.info(f"📈 Feature Engineering Complete:")
        logger.info(f"  Features: {len(X.columns)}")
        logger.info(f"  Samples: {len(X):,}")
        logger.info(f"  Fraud Rate: {y.mean():.4f} ({y.sum():,} frauds)")
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Step 4: Evaluate Results
        logger.info("\n📊 STEP 4: Model Performance Results")
        logger.info("-" * 50)
        
        best_auc = 0
        best_model = None
        
        for model_name, metrics in results.items():
            auc = metrics['auc']
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
            
            if 'f1_score' in metrics:
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = model_name
            
            logger.info("")
        
        # Final Summary
        logger.info("=" * 80)
        logger.info("🎉 SYNAPSE AI FRAUD DETECTION - PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📊 Training Summary:")
        logger.info(f"  Total Datasets Used: {len(all_datasets)}")
        logger.info(f"  Total Training Samples: {len(X):,}")
        logger.info(f"  Features Engineered: {len(X.columns)}")
        logger.info(f"  Models Trained: {len(results)}")
        logger.info("")
        logger.info(f"🏆 Best Performing Model: {best_model.upper()}")
        logger.info(f"  Best AUC Score: {best_auc:.4f}")
        logger.info("")
        logger.info(f"💾 Models Saved To: {trainer.model_dir}")
        logger.info("=" * 80)
        
        # Performance Assessment
        if best_auc > 0.95:
            logger.info("🌟 EXCELLENT! Models exceed target AUC > 0.95")
        elif best_auc > 0.90:
            logger.info("✅ GOOD! Models achieve strong performance > 0.90 AUC")
        elif best_auc > 0.85:
            logger.info("👍 ACCEPTABLE! Models show decent performance > 0.85 AUC")
        else:
            logger.info("⚠️  Models may need additional tuning for production use")
        
        logger.info("\n🚀 Ready for deployment to Synapse AI platform!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("Your fraud detection AI is now trained and ready!")
    else:
        print("\n❌ Pipeline failed. Check logs above for details.")
        sys.exit(1)
