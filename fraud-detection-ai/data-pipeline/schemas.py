from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Address(BaseModel):
    """Address model for billing and shipping information."""
    street: str
    city: str
    state: str
    country: str
    postal_code: str


class TransactionEvent(BaseModel):
    """Raw transaction event from the payment system."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp in UTC")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., regex="^[A-Z]{3}$", description="ISO 4217 currency code")
    merchant_id: str = Field(..., description="Unique merchant identifier")
    user_id: str = Field(..., description="Customer identifier")
    billing_address: Address
    shipping_address: Address
    ip_address: str = Field(..., description="User's IP address")
    user_agent: str = Field(..., description="Browser user agent string")
    device_fingerprint: str = Field(..., description="Unique device identifier hash")


class FeatureVector(BaseModel):
    """Computed features for fraud detection model."""
    transaction_id: str
    
    # Velocity features
    velocity_1m: int = Field(default=0, description="Transactions in last 1 minute")
    velocity_5m: int = Field(default=0, description="Transactions in last 5 minutes") 
    velocity_1h: int = Field(default=0, description="Transactions in last 1 hour")
    velocity_24h: int = Field(default=0, description="Transactions in last 24 hours")
    velocity_7d: int = Field(default=0, description="Transactions in last 7 days")
    
    # Behavioral deviation features
    amount_zscore: float = Field(default=0.0, description="Z-score of amount vs user history")
    amount_mean_7d: float = Field(default=0.0, description="User's 7-day average amount")
    amount_std_7d: float = Field(default=0.0, description="User's 7-day amount std dev")
    
    # Geospatial features
    distance_last_transaction: float = Field(default=0.0, description="Distance from last transaction (km)")
    distance_avg_5_transactions: float = Field(default=0.0, description="Avg distance from last 5 transactions (km)")
    new_country: bool = Field(default=False, description="Transaction from new country")
    new_city: bool = Field(default=False, description="Transaction from new city")
    
    # Device & network features
    unique_devices_24h: int = Field(default=0, description="Unique devices in 24h")
    is_new_ip: bool = Field(default=False, description="New IP address flag")
    is_new_user_agent: bool = Field(default=False, description="New user agent flag")
    is_new_device: bool = Field(default=False, description="New device fingerprint flag")
    
    # Merchant features
    merchant_transaction_count: int = Field(default=0, description="User's transactions with this merchant")
    merchant_first_transaction: bool = Field(default=False, description="First transaction with merchant")


class DecisionType(str, Enum):
    """Fraud detection decision types."""
    APPROVE = "APPROVE"
    DENY = "DENY"
    REVIEW = "REVIEW"


class FraudScore(BaseModel):
    """Fraud detection model output."""
    transaction_id: str
    decision: DecisionType
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score between 0 and 1")
    model_score: float = Field(..., ge=0.0, le=1.0, description="Raw model probability")
    rules_triggered: list[str] = Field(default_factory=list, description="Business rules that fired")
    explanation: Dict[str, Any] = Field(default_factory=dict, description="SHAP feature explanations")


class FeedbackLabel(BaseModel):
    """Analyst feedback for model training."""
    transaction_id: str
    is_fraud: bool
    analyst_id: str
    timestamp: datetime
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analyst confidence in label")
    notes: Optional[str] = None
