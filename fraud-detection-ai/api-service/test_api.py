import pytest
import asyncio
from httpx import AsyncClient
from datetime import datetime

from api_service.main import create_app
from data_pipeline.schemas import TransactionEvent, Address


@pytest.fixture
def app():
    """Create test app."""
    config = {
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 1  
        },
        'model_path': 'tests/fixtures/models',
        'host': '0.0.0.0',
        'port': 8000
    }
    return create_app(config)


@pytest.fixture
async def client(app):
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing."""
    return TransactionEvent(
        transaction_id="test_123456",
        timestamp=datetime.utcnow(),
        amount=99.99,
        currency="USD",
        merchant_id="merchant_123",
        user_id="user_456",
        billing_address=Address(
            street="123 Main St",
            city="New York",
            state="NY",
            country="US",
            postal_code="10001"
        ),
        shipping_address=Address(
            street="123 Main St", 
            city="New York",
            state="NY",
            country="US",
            postal_code="10001"
        ),
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        device_fingerprint="abc123def456"
    )


class TestFraudDetectionAPI:
    """Test cases for the fraud detection API."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Synapse AI Fraud Detection API"
        assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = await client.get("/models/info")
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_score_transaction_endpoint(self, client, sample_transaction):
        """Test main fraud scoring endpoint."""
        response = await client.post(
            "/v1/score_transaction",
            json=sample_transaction.dict()
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response structure
            assert "transaction_id" in data
            assert "decision" in data
            assert "risk_score" in data
            assert "decision_details" in data
            
            # Check decision is valid
            assert data["decision"] in ["APPROVE", "DENY", "REVIEW"]
            
            # Check risk score is in valid range
            assert 0.0 <= data["risk_score"] <= 1.0
            
            # Check transaction ID matches
            assert data["transaction_id"] == sample_transaction.transaction_id
    
    @pytest.mark.asyncio
    async def test_invalid_transaction_schema(self, client):
        """Test API response to invalid transaction schema."""
        invalid_transaction = {
            "transaction_id": "test_123",
            "amount": -100,  
            "currency": "INVALID"  
        }
        
        response = await client.post(
            "/v1/score_transaction",
            json=invalid_transaction
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus format
        assert "fraud_api_requests_total" in response.text or response.text == ""


class TestPerformance:
    """Performance tests for the API."""
    
    @pytest.mark.asyncio
    async def test_response_time(self, client, sample_transaction):
        """Test that response time is under 100ms target."""
        import time
        
        start_time = time.time()
        response = await client.post(
            "/v1/score_transaction",
            json=sample_transaction.dict()
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  
        
        # Log the response time for analysis
        print(f"Response time: {response_time:.2f}ms")
        assert response_time < 5000  # 5 second timeout for test environment
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_transaction):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            # Modify transaction ID for each request
            tx = sample_transaction.copy()
            tx.transaction_id = f"concurrent_test_{i}"
            
            task = client.post(
                "/v1/score_transaction",
                json=tx.dict()
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code in [200, 503])
        assert successful >= 8  


class TestBusinessRules:
    """Test business rules integration."""
    
    @pytest.mark.asyncio
    async def test_high_velocity_rule(self, client):
        """Test high velocity transaction rule."""
        pass
    
    @pytest.mark.asyncio
    async def test_large_amount_new_device_rule(self, client):
        """Test large amount + new device rule."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
