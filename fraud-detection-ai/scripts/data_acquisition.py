"""
Multi-Dataset Fraud Detection Training Pipeline
Combines multiple fraud detection datasets for comprehensive model training
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDatasetManager:
    """Manages multiple fraud detection datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        
    def download_credit_card_dataset(self) -> bool:
        """Download the famous Credit Card Fraud Detection dataset"""
        logger.info("Downloading Credit Card Fraud Detection dataset...")
        
        # This dataset requires Kaggle API - providing instructions
        dataset_info = {
            'name': 'creditcard_fraud',
            'source': 'kaggle',
            'kaggle_dataset': 'mlg-ulb/creditcardfraud',
            'filename': 'creditcard.csv',
            'size': '150MB',
            'transactions': 284807,
            'fraud_rate': 0.172
        }
        
        logger.info(f"Dataset info: {dataset_info}")
        logger.info("To download this dataset, please:")
        logger.info("1. Install Kaggle API: pip install kaggle")
        logger.info("2. Setup Kaggle credentials: ~/.kaggle/kaggle.json")
        logger.info("3. Run: kaggle datasets download -d mlg-ulb/creditcardfraud")
        
        # Check if dataset already exists
        csv_path = self.data_dir / "creditcard.csv"
        if csv_path.exists():
            logger.info("Credit card dataset found locally!")
            return True
        
        return False
    
    def download_ieee_fraud_dataset(self) -> bool:
        """Download IEEE-CIS Fraud Detection dataset"""
        logger.info("Downloading IEEE-CIS Fraud Detection dataset...")
        
        dataset_info = {
            'name': 'ieee_fraud',
            'source': 'kaggle',
            'kaggle_dataset': 'c/ieee-fraud-detection',
            'files': ['train_transaction.csv', 'train_identity.csv'],
            'size': '1.2GB',
            'transactions': 590540,
            'fraud_rate': 3.5
        }
        
        logger.info(f"Dataset info: {dataset_info}")
        logger.info("To download IEEE dataset:")
        logger.info("kaggle competitions download -c ieee-fraud-detection")
        
        # Check if dataset exists
        train_path = self.data_dir / "train_transaction.csv"
        if train_path.exists():
            logger.info("IEEE fraud dataset found locally!")
            return True
        
        return False
    
    def create_synthetic_dataset(self, n_samples: int = 100000) -> pd.DataFrame:
        """Create synthetic fraud detection dataset"""
        logger.info(f"Creating synthetic dataset with {n_samples} samples...")
        
        np.random.seed(42)
        
        # Generate customer profiles
        n_customers = min(10000, n_samples // 10)
        n_merchants = min(5000, n_samples // 20)
        
        # Customer features
        customers = pd.DataFrame({
            'customer_id': range(n_customers),
            'customer_age': np.random.normal(35, 12, n_customers).clip(18, 80).astype(int),
            'customer_income': np.random.lognormal(10, 0.5, n_customers).clip(20000, 500000),
            'customer_risk_score': np.random.beta(2, 5, n_customers),
            'account_age_days': np.random.exponential(365, n_customers).clip(1, 3650).astype(int)
        })
        
        # Generate transactions
        transactions = []
        
        for i in range(n_samples):
            # Basic transaction features
            customer_id = np.random.choice(n_customers)
            merchant_id = np.random.choice(n_merchants)
            
            # Time features
            hour = np.random.choice(24, p=self._get_hourly_distribution())
            day_of_week = np.random.choice(7)
            
            # Amount based on time and customer
            base_amount = np.random.lognormal(3, 1)
            if hour < 6 or hour > 22:  # Suspicious hours
                base_amount *= np.random.uniform(0.5, 3.0)
            
            # Transaction features
            transaction = {
                'transaction_id': f'txn_{i:08d}',
                'customer_id': customer_id,
                'merchant_id': merchant_id,
                'amount': round(base_amount, 2),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'merchant_category': np.random.choice(['retail', 'gas', 'grocery', 'restaurant', 'online']),
                'payment_method': np.random.choice(['credit', 'debit', 'digital_wallet'], p=[0.6, 0.3, 0.1]),
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.7, 0.25, 0.05])
            }
            
            # Velocity features (simplified)
            transaction['customer_tx_count_1h'] = np.random.poisson(2)
            transaction['customer_tx_count_24h'] = np.random.poisson(8)
            transaction['merchant_tx_count_1h'] = np.random.poisson(50)
            
            # Geographic features
            transaction['distance_from_home'] = np.random.exponential(10)
            transaction['new_merchant'] = np.random.choice([0, 1], p=[0.8, 0.2])
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Generate fraud labels using realistic patterns
        df['is_fraud'] = self._generate_fraud_labels(df)
        
        fraud_rate = df['is_fraud'].mean() * 100
        logger.info(f"Synthetic dataset created: {len(df)} transactions, {fraud_rate:.2f}% fraud rate")
        
        return df
    
    def _get_hourly_distribution(self) -> np.ndarray:
        """Get realistic hourly transaction distribution"""
        # More transactions during business hours
        probs = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5
                         0.05, 0.07, 0.08, 0.09, 0.08, 0.07,  # 6-11
                         0.08, 0.09, 0.08, 0.07, 0.06, 0.06,  # 12-17
                         0.05, 0.04, 0.03, 0.02, 0.02, 0.01]) # 18-23
        return probs / probs.sum()
    
    def _generate_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic fraud labels based on transaction patterns"""
        fraud_score = np.zeros(len(df))
        
        # High amount transactions
        fraud_score += (df['amount'] > 1000) * 0.3
        fraud_score += (df['amount'] > 5000) * 0.5
        
        # Unusual hours
        fraud_score += ((df['hour'] < 6) | (df['hour'] > 22)) * 0.2
        
        # High velocity
        fraud_score += (df['customer_tx_count_1h'] > 5) * 0.3
        fraud_score += (df['customer_tx_count_24h'] > 20) * 0.2
        
        # Geographic anomalies
        fraud_score += (df['distance_from_home'] > 100) * 0.4
        fraud_score += df['new_merchant'] * 0.1
        
        # Digital wallet + high amount
        fraud_score += ((df['payment_method'] == 'digital_wallet') & (df['amount'] > 500)) * 0.3
        
        # Random fraud (represents unknown patterns)
        fraud_score += np.random.random(len(df)) * 0.1
        
        # Convert to binary with some noise
        fraud_probability = 1 / (1 + np.exp(-3 * (fraud_score - 0.8)))  # Sigmoid
        is_fraud = np.random.random(len(df)) < fraud_probability
        
        return is_fraud.astype(int)
    
    def create_paysim_style_dataset(self, n_samples: int = 50000) -> pd.DataFrame:
        """Create PaySim-style mobile money dataset"""
        logger.info(f"Creating PaySim-style dataset with {n_samples} samples...")
        
        np.random.seed(123)
        
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        type_probs = [0.4, 0.2, 0.2, 0.1, 0.1]
        
        transactions = []
        
        for i in range(n_samples):
            tx_type = np.random.choice(transaction_types, p=type_probs)
            
            # Generate amounts based on transaction type
            if tx_type == 'PAYMENT':
                amount = np.random.lognormal(4, 1)
            elif tx_type in ['TRANSFER', 'CASH_OUT']:
                amount = np.random.lognormal(6, 1.5)
            else:
                amount = np.random.lognormal(5, 1)
            
            transaction = {
                'step': i // 1000,  # Time step
                'type': tx_type,
                'amount': round(amount, 2),
                'nameOrig': f'C{np.random.randint(1, 10000)}',
                'oldbalanceOrg': np.random.exponential(10000),
                'nameDest': f'C{np.random.randint(1, 10000)}',
                'oldbalanceDest': np.random.exponential(10000),
            }
            
            # Calculate new balances
            transaction['newbalanceOrig'] = max(0, transaction['oldbalanceOrg'] - amount)
            transaction['newbalanceDest'] = transaction['oldbalanceDest'] + amount
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Generate fraud labels for PaySim-style patterns
        df['isFraud'] = self._generate_paysim_fraud_labels(df)
        
        fraud_rate = df['isFraud'].mean() * 100
        logger.info(f"PaySim-style dataset created: {len(df)} transactions, {fraud_rate:.2f}% fraud rate")
        
        return df
    
    def _generate_paysim_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate fraud labels for PaySim-style data"""
        fraud_score = np.zeros(len(df))
        
        # TRANSFER and CASH_OUT are more likely to be fraudulent
        fraud_score += (df['type'].isin(['TRANSFER', 'CASH_OUT'])) * 0.4
        
        # Large amounts
        fraud_score += (df['amount'] > 200000) * 0.6
        
        # Zero balance after transaction (possible account takeover)
        fraud_score += (df['newbalanceOrig'] == 0) * 0.3
        fraud_score += (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0) * 0.4
        
        # Destination account with zero initial balance receiving large amount
        fraud_score += (df['oldbalanceDest'] == 0) & (df['amount'] > 100000) * 0.5
        
        # Convert to binary
        fraud_probability = 1 / (1 + np.exp(-4 * (fraud_score - 0.9)))
        is_fraud = np.random.random(len(df)) < fraud_probability
        
        return is_fraud.astype(int)
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        datasets = {}
        
        # Try to load Credit Card dataset
        creditcard_path = self.data_dir / "creditcard.csv"
        if creditcard_path.exists():
            logger.info("Loading Credit Card Fraud dataset...")
            datasets['creditcard'] = pd.read_csv(creditcard_path)
            logger.info(f"Loaded {len(datasets['creditcard'])} credit card transactions")
        
        # Try to load IEEE dataset
        ieee_path = self.data_dir / "train_transaction.csv"
        if ieee_path.exists():
            logger.info("Loading IEEE Fraud dataset...")
            datasets['ieee'] = pd.read_csv(ieee_path)
            logger.info(f"Loaded {len(datasets['ieee'])} IEEE transactions")
        
        # Always create synthetic datasets
        datasets['synthetic'] = self.create_synthetic_dataset(100000)
        datasets['paysim'] = self.create_paysim_style_dataset(50000)
        
        total_transactions = sum(len(df) for df in datasets.values())
        logger.info(f"Total datasets loaded: {len(datasets)}")
        logger.info(f"Total transactions available: {total_transactions:,}")
        
        return datasets
    
    def create_e_commerce_dataset(self, n_samples: int = 75000) -> pd.DataFrame:
        """Create e-commerce specific fraud dataset"""
        logger.info(f"Creating e-commerce fraud dataset with {n_samples} samples...")
        
        np.random.seed(456)
        
        products = ['electronics', 'clothing', 'books', 'home', 'beauty', 'sports', 'toys']
        browsers = ['chrome', 'firefox', 'safari', 'edge', 'other']
        countries = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU', 'BR', 'IN', 'MX']
        
        transactions = []
        
        for i in range(n_samples):
            # Customer session data
            session_length = np.random.exponential(300)  # seconds
            pages_visited = np.random.poisson(5) + 1
            
            transaction = {
                'transaction_id': f'ecom_{i:08d}',
                'user_id': f'user_{np.random.randint(1, 20000)}',
                'session_id': f'sess_{np.random.randint(1, 100000)}',
                'product_category': np.random.choice(products),
                'product_price': round(np.random.lognormal(4, 1), 2),
                'quantity': np.random.choice([1, 2, 3, 4, 5], p=[0.7, 0.15, 0.1, 0.03, 0.02]),
                'browser': np.random.choice(browsers, p=[0.6, 0.15, 0.15, 0.08, 0.02]),
                'country': np.random.choice(countries, p=[0.4, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.08, 0.08]),
                'session_length_sec': round(session_length),
                'pages_visited': pages_visited,
                'is_mobile': np.random.choice([0, 1], p=[0.4, 0.6]),
                'hour_of_day': np.random.choice(24),
                'day_of_week': np.random.choice(7),
                'email_domain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'other'], 
                                               p=[0.4, 0.2, 0.15, 0.15, 0.1]),
                'account_age_days': np.random.exponential(200),
                'previous_purchases': np.random.poisson(3),
                'shipping_address_matches_billing': np.random.choice([0, 1], p=[0.1, 0.9])
            }
            
            # Calculate total amount
            transaction['total_amount'] = round(transaction['product_price'] * transaction['quantity'], 2)
            
            # Behavioral features
            transaction['time_on_product_page'] = np.random.exponential(60)
            transaction['cart_abandonment_count'] = np.random.poisson(1)
            transaction['uses_vpn'] = np.random.choice([0, 1], p=[0.95, 0.05])
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Generate e-commerce fraud labels
        df['is_fraud'] = self._generate_ecommerce_fraud_labels(df)
        
        fraud_rate = df['is_fraud'].mean() * 100
        logger.info(f"E-commerce dataset created: {len(df)} transactions, {fraud_rate:.2f}% fraud rate")
        
        return df
    
    def _generate_ecommerce_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate fraud labels for e-commerce data"""
        fraud_score = np.zeros(len(df))
        
        # High-value electronics purchases
        fraud_score += ((df['product_category'] == 'electronics') & (df['total_amount'] > 1000)) * 0.4
        
        # Multiple high-value items
        fraud_score += ((df['quantity'] > 3) & (df['total_amount'] > 500)) * 0.3
        
        # New accounts with high purchases
        fraud_score += ((df['account_age_days'] < 30) & (df['total_amount'] > 300)) * 0.5
        
        # Suspicious browsing behavior
        fraud_score += (df['session_length_sec'] < 60) & (df['total_amount'] > 200) * 0.3
        fraud_score += (df['pages_visited'] == 1) & (df['total_amount'] > 100) * 0.2
        
        # Address mismatch
        fraud_score += (df['shipping_address_matches_billing'] == 0) * 0.3
        
        # VPN usage
        fraud_score += df['uses_vpn'] * 0.4
        
        # Unusual time patterns
        fraud_score += ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 23)) * 0.2
        
        # No previous purchases + high amount
        fraud_score += ((df['previous_purchases'] == 0) & (df['total_amount'] > 200)) * 0.4
        
        # Convert to binary
        fraud_probability = 1 / (1 + np.exp(-3.5 * (fraud_score - 0.85)))
        is_fraud = np.random.random(len(df)) < fraud_probability
        
        return is_fraud.astype(int)
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Save all datasets to files"""
        save_dir = self.data_dir / "processed"
        save_dir.mkdir(exist_ok=True)
        
        for name, df in datasets.items():
            filepath = save_dir / f"{name}_fraud_data.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} dataset to {filepath}")
            
            # Print dataset summary
            fraud_count = df.iloc[:, -1].sum()  # Assuming last column is fraud label
            fraud_rate = (fraud_count / len(df)) * 100
            logger.info(f"{name}: {len(df):,} transactions, {fraud_count:,} frauds ({fraud_rate:.2f}%)")


def main():
    """Main function to acquire and prepare all fraud datasets"""
    logger.info("🚀 Starting Multi-Dataset Fraud Detection Data Acquisition")
    logger.info("=" * 70)
    
    # Initialize dataset manager
    manager = FraudDatasetManager()
    
    # Check for external datasets
    manager.download_credit_card_dataset()
    manager.download_ieee_fraud_dataset()
    
    # Load all available datasets
    datasets = manager.load_all_datasets()
    
    # Create additional specialized datasets
    datasets['ecommerce'] = manager.create_e_commerce_dataset(75000)
    
    # Save all datasets
    manager.save_datasets(datasets)
    
    # Print summary
    logger.info("=" * 70)
    logger.info("📊 DATASET ACQUISITION SUMMARY")
    logger.info("=" * 70)
    
    total_transactions = 0
    total_frauds = 0
    
    for name, df in datasets.items():
        fraud_col = df.columns[-1]  # Last column is fraud indicator
        fraud_count = df[fraud_col].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        
        logger.info(f"{name.upper():12} | {len(df):8,} txns | {fraud_count:6,} frauds | {fraud_rate:5.2f}%")
        
        total_transactions += len(df)
        total_frauds += fraud_count
    
    overall_fraud_rate = (total_frauds / total_transactions) * 100
    
    logger.info("=" * 70)
    logger.info(f"{'TOTAL':12} | {total_transactions:8,} txns | {total_frauds:6,} frauds | {overall_fraud_rate:5.2f}%")
    logger.info("=" * 70)
    
    logger.info("✅ All datasets prepared! Ready for model training.")
    logger.info("📁 Datasets saved in: data/processed/")
    
    return datasets


if __name__ == "__main__":
    datasets = main()
