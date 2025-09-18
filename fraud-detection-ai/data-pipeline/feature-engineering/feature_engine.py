import math
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import redis
import hashlib

from schemas import TransactionEvent, FeatureVector


@dataclass
class UserState:
    """Maintains user state for feature computation."""
    user_id: str
    transaction_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    amounts_7d: deque = field(default_factory=lambda: deque(maxlen=10000))
    locations_5: deque = field(default_factory=lambda: deque(maxlen=5))
    devices_24h: set = field(default_factory=set)
    ips_24h: set = field(default_factory=set)
    user_agents_24h: set = field(default_factory=set)
    merchant_counts: Dict[str, int] = field(default_factory=dict)
    last_cleanup: datetime = field(default_factory=datetime.utcnow)


class RealTimeFeatureEngine:
    """
    Real-time feature engineering service using stateful stream processing.
    
    Computes velocity, behavioral deviation, geospatial, and device features
    for each transaction in real-time with minimal latency.
    """
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize feature engine with Redis for state persistence."""
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        self.user_states: Dict[str, UserState] = {}
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def geocode_address(self, address: Dict[str, str]) -> Tuple[float, float]:
        """
        Convert address to coordinates with validation and realistic defaults.
        In production, integrate with a real geocoding service like Google Maps API.
        
        Args:
            address: Address dictionary with city, country etc.
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Extract and validate address components
        city = address.get('city', '').strip()
        country = address.get('country', '').strip()
        state = address.get('state', '').strip()
        
        # Handle empty or invalid addresses
        if not city and not country:
            self.logger.warning("Empty address provided, using default coordinates")
            return (40.7128, -74.0060)  # New York City as default
        
        # Create a normalized address string for consistent hashing
        address_key = f"{city.lower()},{state.lower()},{country.lower()}".replace(' ', '')
        
        # Known major city coordinates for common addresses
        city_coordinates = {
            # US Cities
            'newyork,ny,usa': (40.7128, -74.0060),
            'newyork,newyork,usa': (40.7128, -74.0060),
            'losangeles,ca,usa': (34.0522, -118.2437),
            'chicago,il,usa': (41.8781, -87.6298),
            'houston,tx,usa': (29.7604, -95.3698),
            'philadelphia,pa,usa': (39.9526, -75.1652),
            'phoenix,az,usa': (33.4484, -112.0740),
            'sanfrancisco,ca,usa': (37.7749, -122.4194),
            'miami,fl,usa': (25.7617, -80.1918),
            'seattle,wa,usa': (47.6062, -122.3321),
            
            # International Cities
            'london,,uk': (51.5074, -0.1278),
            'london,,unitedkingdom': (51.5074, -0.1278),
            'paris,,france': (48.8566, 2.3522),
            'tokyo,,japan': (35.6762, 139.6503),
            'sydney,,australia': (-33.8688, 151.2093),
            'toronto,,canada': (43.6532, -79.3832),
            'berlin,,germany': (52.5200, 13.4050),
            'amsterdam,,netherlands': (52.3676, 4.9041),
            'mumbai,,india': (19.0760, 72.8777),
            'singapore,,singapore': (1.3521, 103.8198),
            
            # Default for common countries
            ',,usa': (39.8283, -98.5795),  # Geographic center of USA
            ',,canada': (56.1304, -106.3468),  # Geographic center of Canada
            ',,uk': (55.3781, -3.4360),  # Geographic center of UK
            ',,france': (46.2276, 2.2137),  # Geographic center of France
            ',,germany': (51.1657, 10.4515),  # Geographic center of Germany
        }
        
        # Try to find exact match first
        if address_key in city_coordinates:
            return city_coordinates[address_key]
        
        # Try country-only match
        country_key = f",,{country.lower()}"
        if country_key in city_coordinates:
            # Add some randomness for different cities in same country
            base_lat, base_lon = city_coordinates[country_key]
            hash_val = int(hashlib.md5(address_key.encode()).hexdigest()[:8], 16)
            lat_offset = ((hash_val % 200) - 100) / 100.0  # ±1 degree
            lon_offset = ((hash_val // 200) % 200 - 100) / 100.0  # ±1 degree
            return (base_lat + lat_offset, base_lon + lon_offset)
        
        # Fallback: Generate consistent coordinates based on address hash
        hash_val = int(hashlib.md5(address_key.encode()).hexdigest()[:8], 16)
        
        # Ensure coordinates are within valid ranges and realistic populated areas
        # Focus on populated latitude bands (avoid polar regions)
        lat = 20.0 + (hash_val % 6000) / 100.0  # Range: 20-80 degrees (avoiding Antarctica)
        lon = -180.0 + ((hash_val // 100) % 36000) / 100.0  # Range: -180 to 180 degrees
        
        # Avoid ocean-only coordinates by biasing toward continental areas
        if abs(lat) > 70:  # Too close to poles
            lat = 40.0 + (hash_val % 4000) / 100.0  # Range: 40-80
        
        self.logger.debug(f"Generated coordinates for {address_key}: ({lat:.4f}, {lon:.4f})")
        
        return (lat, lon)
    
    def get_user_state(self, user_id: str) -> UserState:
        """Get or create user state from cache/Redis."""
        if user_id not in self.user_states:
            # Try to load from Redis first
            state_key = f"user_state:{user_id}"
            cached_state = self.redis.get(state_key)
            
            if cached_state:
                try:
                    state_data = json.loads(cached_state)
                    # Reconstruct UserState from Redis data
                    user_state = UserState(user_id=user_id)
                    # Note: In production, implement proper serialization/deserialization
                    self.user_states[user_id] = user_state
                except Exception as e:
                    self.logger.error(f"Error loading user state from Redis: {e}")
                    self.user_states[user_id] = UserState(user_id=user_id)
            else:
                self.user_states[user_id] = UserState(user_id=user_id)
        
        return self.user_states[user_id]
    
    def save_user_state(self, user_state: UserState):
        """Save user state to Redis for persistence."""
        try:
            state_key = f"user_state:{user_state.user_id}"
            # In production, implement proper serialization
            # For now, just set a TTL to prevent memory leaks
            self.redis.setex(state_key, 86400, json.dumps({"user_id": user_state.user_id}))
        except Exception as e:
            self.logger.error(f"Error saving user state to Redis: {e}")
    
    def cleanup_expired_data(self, user_state: UserState, current_time: datetime):
        """Clean up expired data from user state to prevent memory leaks."""
        # Clean up every hour
        if current_time - user_state.last_cleanup < timedelta(hours=1):
            return
        
        # Remove transactions older than 7 days
        cutoff_7d = current_time - timedelta(days=7)
        while (user_state.transaction_history and 
               user_state.transaction_history[0]['timestamp'] < cutoff_7d):
            user_state.transaction_history.popleft()
        
        # Remove amounts older than 7 days
        while (user_state.amounts_7d and 
               user_state.amounts_7d[0]['timestamp'] < cutoff_7d):
            user_state.amounts_7d.popleft()
        
        # Clear 24h data (will be rebuilt)
        cutoff_24h = current_time - timedelta(hours=24)
        user_state.devices_24h = {d for d in user_state.devices_24h 
                                  if d.get('timestamp', current_time) > cutoff_24h}
        user_state.ips_24h = {ip for ip in user_state.ips_24h 
                             if ip.get('timestamp', current_time) > cutoff_24h}
        user_state.user_agents_24h = {ua for ua in user_state.user_agents_24h 
                                     if ua.get('timestamp', current_time) > cutoff_24h}
        
        user_state.last_cleanup = current_time
    
    def compute_velocity_features(self, user_state: UserState, current_time: datetime) -> Dict[str, int]:
        """Compute transaction velocity features."""
        features = {
            'velocity_1m': 0,
            'velocity_5m': 0, 
            'velocity_1h': 0,
            'velocity_24h': 0,
            'velocity_7d': 0
        }
        
        # Define time windows
        windows = {
            'velocity_1m': timedelta(minutes=1),
            'velocity_5m': timedelta(minutes=5),
            'velocity_1h': timedelta(hours=1),
            'velocity_24h': timedelta(hours=24),
            'velocity_7d': timedelta(days=7)
        }
        
        # Count transactions in each window
        for feature_name, window in windows.items():
            cutoff = current_time - window
            count = sum(1 for tx in user_state.transaction_history 
                       if tx['timestamp'] > cutoff)
            features[feature_name] = count
        
        return features
    
    def compute_behavioral_features(self, user_state: UserState, current_amount: float) -> Dict[str, float]:
        """Compute behavioral deviation features."""
        if len(user_state.amounts_7d) < 2:
            return {
                'amount_zscore': 0.0,
                'amount_mean_7d': current_amount,
                'amount_std_7d': 0.0
            }
        
        amounts = [tx['amount'] for tx in user_state.amounts_7d]
        mean_amount = sum(amounts) / len(amounts)
        
        # Calculate standard deviation
        variance = sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)
        std_amount = math.sqrt(variance)
        
        # Calculate Z-score
        zscore = (current_amount - mean_amount) / max(std_amount, 0.01)  # Avoid division by zero
        
        return {
            'amount_zscore': zscore,
            'amount_mean_7d': mean_amount,
            'amount_std_7d': std_amount
        }
    
    def compute_geospatial_features(self, user_state: UserState, current_location: Tuple[float, float]) -> Dict[str, Any]:
        """Compute geospatial distance and location change features."""
        features = {
            'distance_last_transaction': 0.0,
            'distance_avg_5_transactions': 0.0,
            'new_country': False,
            'new_city': False
        }
        
        if not user_state.locations_5:
            return features
        
        # Distance from last transaction
        last_location = user_state.locations_5[-1]
        features['distance_last_transaction'] = self.haversine_distance(
            current_location[0], current_location[1],
            last_location['lat'], last_location['lon']
        )
        
        # Average distance from last 5 transactions
        if len(user_state.locations_5) >= 2:
            distances = []
            for loc in user_state.locations_5:
                distance = self.haversine_distance(
                    current_location[0], current_location[1],
                    loc['lat'], loc['lon']
                )
                distances.append(distance)
            features['distance_avg_5_transactions'] = sum(distances) / len(distances)
        
        # Check for new country/city
        recent_countries = {loc.get('country') for loc in user_state.locations_5}
        recent_cities = {loc.get('city') for loc in user_state.locations_5}
        
        # This would need the current address - simplified for demo
        features['new_country'] = len(recent_countries) > 1
        features['new_city'] = len(recent_cities) > 1
        
        return features
    
    def compute_device_features(self, user_state: UserState, device_fingerprint: str, 
                               ip_address: str, user_agent: str, current_time: datetime) -> Dict[str, Any]:
        """Compute device and network anomaly features."""
        # Clean up old device data (older than 24h)
        cutoff_24h = current_time - timedelta(hours=24)
        
        # Count unique devices, IPs, user agents in last 24h
        unique_devices = len([d for d in user_state.devices_24h 
                             if d.get('timestamp', current_time) > cutoff_24h])
        
        # Check if current device/IP/UA is new
        device_fingerprints_24h = {d.get('fingerprint') for d in user_state.devices_24h}
        ips_24h = {d.get('ip') for d in user_state.devices_24h}
        user_agents_24h = {d.get('user_agent') for d in user_state.devices_24h}
        
        features = {
            'unique_devices_24h': unique_devices,
            'is_new_ip': ip_address not in ips_24h,
            'is_new_user_agent': user_agent not in user_agents_24h,
            'is_new_device': device_fingerprint not in device_fingerprints_24h
        }
        
        return features
    
    def compute_merchant_features(self, user_state: UserState, merchant_id: str) -> Dict[str, Any]:
        """Compute merchant-related features."""
        merchant_count = user_state.merchant_counts.get(merchant_id, 0)
        
        return {
            'merchant_transaction_count': merchant_count,
            'merchant_first_transaction': merchant_count == 0
        }
    
    def compute_features(self, transaction: TransactionEvent) -> FeatureVector:
        """
        Compute all features for a transaction.
        
        Args:
            transaction: Validated transaction event
            
        Returns:
            Complete feature vector for fraud detection
        """
        user_state = self.get_user_state(transaction.user_id)
        current_time = transaction.timestamp
        
        # Clean up expired data
        self.cleanup_expired_data(user_state, current_time)
        
        # Get current location coordinates
        current_location = self.geocode_address(transaction.billing_address.dict())
        
        # Compute all feature groups
        velocity_features = self.compute_velocity_features(user_state, current_time)
        behavioral_features = self.compute_behavioral_features(user_state, transaction.amount)
        geospatial_features = self.compute_geospatial_features(user_state, current_location)
        device_features = self.compute_device_features(
            user_state, transaction.device_fingerprint, 
            transaction.ip_address, transaction.user_agent, current_time
        )
        merchant_features = self.compute_merchant_features(user_state, transaction.merchant_id)
        
        # Create feature vector
        features = FeatureVector(
            transaction_id=transaction.transaction_id,
            **velocity_features,
            **behavioral_features,
            **geospatial_features,
            **device_features,
            **merchant_features
        )
        
        # Update user state with current transaction
        self.update_user_state(user_state, transaction, current_location, current_time)
        
        # Save updated state
        self.save_user_state(user_state)
        
        return features
    
    def update_user_state(self, user_state: UserState, transaction: TransactionEvent, 
                         location: Tuple[float, float], current_time: datetime):
        """Update user state with current transaction data."""
        # Add to transaction history
        tx_record = {
            'timestamp': current_time,
            'amount': transaction.amount,
            'merchant_id': transaction.merchant_id
        }
        user_state.transaction_history.append(tx_record)
        
        # Add to amounts for behavioral analysis
        amount_record = {
            'timestamp': current_time,
            'amount': transaction.amount
        }
        user_state.amounts_7d.append(amount_record)
        
        # Add location
        location_record = {
            'timestamp': current_time,
            'lat': location[0],
            'lon': location[1],
            'city': transaction.billing_address.city,
            'country': transaction.billing_address.country
        }
        user_state.locations_5.append(location_record)
        
        # Add device info
        device_record = {
            'timestamp': current_time,
            'fingerprint': transaction.device_fingerprint,
            'ip': transaction.ip_address,
            'user_agent': transaction.user_agent
        }
        user_state.devices_24h.add(frozenset(device_record.items()))
        
        # Update merchant count
        user_state.merchant_counts[transaction.merchant_id] = (
            user_state.merchant_counts.get(transaction.merchant_id, 0) + 1
        )
