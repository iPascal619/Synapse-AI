import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError
from datetime import datetime

from schemas import TransactionEvent


class TransactionIngestionService:
    """
    High-throughput transaction event ingestion service using Apache Kafka.
    
    Handles JSON schema validation, event preprocessing, and routing to
    the feature engineering pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Kafka consumer and producer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kafka consumer for incoming transactions
        self.consumer = KafkaConsumer(
            self.config['kafka']['input_topic'],
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            group_id=self.config['kafka']['consumer_group_id'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=1000,  # High throughput
            fetch_min_bytes=1024,   # Batch fetching
        )
        
        # Kafka producer for validated events
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='snappy',  # Compression for throughput
            batch_size=16384,           # Larger batches
            linger_ms=10,               # Small delay for batching
        )
        
        self.processed_count = 0
        self.error_count = 0
    
    def validate_transaction_event(self, raw_event: Dict[str, Any]) -> Optional[TransactionEvent]:
        """
        Validate incoming transaction event against Pydantic schema.
        
        Args:
            raw_event: Raw JSON event from Kafka
            
        Returns:
            Validated TransactionEvent or None if invalid
        """
        try:
            # Parse timestamp if it's a string
            if isinstance(raw_event.get('timestamp'), str):
                raw_event['timestamp'] = datetime.fromisoformat(
                    raw_event['timestamp'].replace('Z', '+00:00')
                )
            
            transaction = TransactionEvent(**raw_event)
            return transaction
            
        except ValidationError as e:
            self.logger.error(f"Schema validation failed for transaction {raw_event.get('transaction_id', 'unknown')}: {e}")
            self.error_count += 1
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error validating transaction: {e}")
            self.error_count += 1
            return None
    
    def enrich_transaction(self, transaction: TransactionEvent) -> Dict[str, Any]:
        """
        Enrich transaction with additional metadata for processing.
        
        Args:
            transaction: Validated transaction event
            
        Returns:
            Enriched transaction data
        """
        enriched = transaction.dict()
        
        # Add processing metadata
        enriched['ingestion_timestamp'] = datetime.utcnow().isoformat()
        enriched['processing_version'] = self.config.get('processing_version', '1.0')
        
        # Extract additional fields for feature engineering
        enriched['hour_of_day'] = transaction.timestamp.hour
        enriched['day_of_week'] = transaction.timestamp.weekday()
        enriched['is_weekend'] = transaction.timestamp.weekday() >= 5
        
        return enriched
    
    def process_events(self):
        """
        Main event processing loop.
        
        Continuously consumes events from Kafka, validates them,
        and forwards to feature engineering pipeline.
        """
        self.logger.info("Starting transaction ingestion service...")
        
        try:
            for message in self.consumer:
                try:
                    raw_event = message.value
                    
                    # Validate transaction schema
                    transaction = self.validate_transaction_event(raw_event)
                    if transaction is None:
                        continue
                    
                    # Enrich with metadata
                    enriched_transaction = self.enrich_transaction(transaction)
                    
                    # Forward to feature engineering pipeline
                    self.producer.send(
                        self.config['kafka']['feature_engineering_topic'],
                        value=enriched_transaction
                    )
                    
                    self.processed_count += 1
                    
                    # Log progress every 1000 transactions
                    if self.processed_count % 1000 == 0:
                        self.logger.info(f"Processed {self.processed_count} transactions, {self.error_count} errors")
                        
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    self.error_count += 1
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down ingestion service...")
        except Exception as e:
            self.logger.error(f"Fatal error in ingestion service: {e}")
            raise
        finally:
            self.consumer.close()
            self.producer.close()
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Return service health statistics."""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'status': 'healthy' if self.error_count / max(self.processed_count, 1) < 0.05 else 'degraded'
        }


if __name__ == "__main__":
    # Example configuration
    config = {
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'input_topic': 'transactions',
            'feature_engineering_topic': 'transactions_enriched',
            'consumer_group_id': 'fraud_detection_ingestion'
        },
        'processing_version': '1.0'
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start ingestion service
    service = TransactionIngestionService(config)
    service.process_events()
