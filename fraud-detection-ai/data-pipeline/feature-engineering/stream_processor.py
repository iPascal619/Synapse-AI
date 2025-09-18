import json
import logging
from typing import Dict, Any
from kafka import KafkaConsumer, KafkaProducer
import redis
from datetime import datetime

from feature_engine import RealTimeFeatureEngine
from schemas import TransactionEvent, FeatureVector


class StreamProcessor:
    """
    Apache Flink-style stream processing service for real-time feature computation.
    
    Processes enriched transaction events from Kafka and computes features
    for fraud detection models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize stream processor with Kafka and Redis connections."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kafka consumer for enriched transactions
        self.consumer = KafkaConsumer(
            self.config['kafka']['input_topic'],
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            group_id=self.config['kafka']['consumer_group_id'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=500,  # Moderate batch size for feature computation
        )
        
        # Kafka producer for feature vectors
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            compression_type='snappy',
            acks='all',  # Ensure reliability for feature data
        )
        
        # Redis for state management
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            decode_responses=True
        )
        
        # Feature engine
        self.feature_engine = RealTimeFeatureEngine(self.redis_client)
        
        self.processed_count = 0
        self.error_count = 0
    
    def process_transaction(self, enriched_transaction: Dict[str, Any]) -> FeatureVector:
        """
        Process a single transaction and compute features.
        
        Args:
            enriched_transaction: Transaction event with metadata
            
        Returns:
            Computed feature vector
        """
        try:
            # Convert to TransactionEvent
            transaction = TransactionEvent(**enriched_transaction)
            
            # Compute features using the feature engine
            features = self.feature_engine.compute_features(transaction)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing transaction {enriched_transaction.get('transaction_id', 'unknown')}: {e}")
            raise
    
    def run(self):
        """
        Main stream processing loop.
        
        Continuously processes transactions and computes features in real-time.
        """
        self.logger.info("Starting stream processor for feature engineering...")
        
        try:
            for message in self.consumer:
                try:
                    enriched_transaction = message.value
                    transaction_id = enriched_transaction.get('transaction_id', 'unknown')
                    
                    # Compute features
                    start_time = datetime.utcnow()
                    features = self.process_transaction(enriched_transaction)
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    # Add processing metadata
                    feature_data = features.dict()
                    feature_data['feature_processing_time_ms'] = processing_time
                    feature_data['feature_processing_timestamp'] = datetime.utcnow().isoformat()
                    
                    # Send to ML inference pipeline
                    self.producer.send(
                        self.config['kafka']['output_topic'],
                        value=feature_data
                    )
                    
                    self.processed_count += 1
                    
                    # Log processing metrics
                    if self.processed_count % 100 == 0:
                        self.logger.info(
                            f"Processed {self.processed_count} transactions, "
                            f"avg processing time: {processing_time:.2f}ms, "
                            f"{self.error_count} errors"
                        )
                    
                    # Alert if processing is too slow
                    if processing_time > 50:  # More than 50ms
                        self.logger.warning(
                            f"Slow feature processing for {transaction_id}: {processing_time:.2f}ms"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error in stream processing: {e}")
                    self.error_count += 1
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down stream processor...")
        except Exception as e:
            self.logger.error(f"Fatal error in stream processor: {e}")
            raise
        finally:
            self.consumer.close()
            self.producer.close()
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Return stream processor health statistics."""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'redis_connection': self.redis_client.ping(),
            'status': 'healthy' if self.error_count / max(self.processed_count, 1) < 0.01 else 'degraded'
        }


if __name__ == "__main__":
    # Configuration
    config = {
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'input_topic': 'transactions_enriched',
            'output_topic': 'features',
            'consumer_group_id': 'fraud_detection_features'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start stream processor
    processor = StreamProcessor(config)
    processor.run()
