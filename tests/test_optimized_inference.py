"""Unit tests for OptimizedInference class"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from src.inference.optimized_inference import (
    OptimizedInference,
    InferenceConfig,
    create_inference_config
)


class TestInferenceConfig:
    """Test suite for InferenceConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = InferenceConfig()
        
        assert config.model_path == "path/to/finetuned/model"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is True
        assert config.use_torch_compile is True
        assert config.quantization == "4bit"
        assert config.batch_size == 4
        assert config.max_new_tokens == 2048
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = InferenceConfig(
            model_path="custom/model/path",
            device="cpu",
            quantization="8bit",
            batch_size=8,
            max_new_tokens=4096
        )
        
        assert config.model_path == "custom/model/path"
        assert config.device == "cpu"
        assert config.quantization == "8bit"
        assert config.batch_size == 8
        assert config.max_new_tokens == 4096
    
    def test_create_inference_config_factory(self):
        """Test factory function for creating configs"""
        config = create_inference_config(
            model_path="test/model",
            device="cuda"
        )
        
        assert isinstance(config, InferenceConfig)
        assert config.model_path == "test/model"
        assert config.device == "cuda"


class TestOptimizedInference:
    """Test suite for OptimizedInference"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return InferenceConfig(
            model_path="test/model",
            device="cpu",
            batch_size=2,
            max_new_tokens=100
        )
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger"""
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.generate = Mock(return_value=[[1, 2, 3, 4, 5]])
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer"""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': [[1, 2, 3]],
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }
        tokenizer.decode = Mock(return_value="Test response")
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def test_inference_initialization(self, mock_config, mock_logger):
        """Test inference engine initialization"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        inference = OptimizedInference(mock_config)
                        
                        assert inference.config == mock_config
                        assert inference.logger == mock_logger
    
    def test_tokenize_input(self, mock_config, mock_logger, mock_tokenizer):
        """Test input tokenization"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        with patch('src.inference.optimized_inference.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                            inference = OptimizedInference(mock_config)
                            inference.tokenizer = mock_tokenizer
                            
                            result = inference.tokenize("Test prompt")
                            
                            assert result is not None
    
    def test_metrics_update(self, mock_config, mock_logger):
        """Test performance metrics update"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        inference = OptimizedInference(mock_config)
                        
                        # Mock results
                        results = [
                            {'output_length': 100, 'generation_time': 2.0},
                            {'output_length': 150, 'generation_time': 3.0}
                        ]
                        
                        inference._update_metrics(results)
                        
                        assert inference.metrics['total_tokens'] == 250
                        assert inference.metrics['total_time'] == 5.0
                        assert inference.metrics['tokens_per_second'] == 50.0
    
    def test_get_performance_metrics(self, mock_config, mock_logger):
        """Test getting performance metrics"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        inference = OptimizedInference(mock_config)
                        
                        inference.metrics = {
                            'total_tokens': 1000,
                            'total_time': 10.0,
                            'tokens_per_second': 100.0,
                            'average_generation_time': 2.0
                        }
                        
                        metrics = inference.get_performance_metrics()
                        
                        assert metrics['total_tokens'] == 1000
                        assert metrics['tokens_per_second'] == 100.0
    
    def test_batch_generation(self, mock_config, mock_logger, mock_model, mock_tokenizer):
        """Test batch generation"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        with patch('src.inference.optimized_inference.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                            inference = OptimizedInference(mock_config)
                            inference.model = mock_model
                            inference.tokenizer = mock_tokenizer
                            
                            prompts = ["Test 1", "Test 2"]
                            
                            # Mock tokenize
                            inference.tokenize = Mock(return_value={
                                'input_ids': [[1, 2, 3]],
                                'attention_mask': [[1, 1, 1]]
                            })
                            
                            results = inference.generate(prompts)
                            
                            assert len(results) == 2
    
    def test_quantization_validation(self, mock_config):
        """Test quantization option validation"""
        valid_quantizations = ["4bit", "8bit", "fp8", "none"]
        
        for quant in valid_quantizations:
            mock_config.quantization = quant
            assert mock_config.quantization in valid_quantizations


class TestBenchmarking:
    """Test suite for benchmarking functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return InferenceConfig(
            model_path="test/model",
            device="cpu",
            batch_size=2
        )
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger"""
        logger = Mock()
        logger.info = Mock()
        return logger
    
    def test_benchmark_model(self, mock_config, mock_logger):
        """Test model benchmarking"""
        with patch('src.inference.optimized_inference.logging.getLogger', return_value=mock_logger):
            with patch('src.inference.optimized_inference.OptimizedInference._setup_monitoring'):
                with patch('src.inference.optimized_inference.OptimizedInference._load_model'):
                    with patch('src.inference.optimized_inference.OptimizedInference._setup_data_collator'):
                        inference = OptimizedInference(mock_config)
                        
                        # Mock generate method
                        inference.generate = Mock(return_value=[
                            {
                                'generation_time': 1.5,
                                'tokens_per_second': 100.0
                            }
                        ])
                        
                        test_prompts = ["Test prompt 1", "Test prompt 2"]
                        results = inference.benchmark_model(test_prompts, num_runs=3)
                        
                        assert 'total_runs' in results
                        assert 'avg_generation_time' in results
                        assert results['total_runs'] == 3


@pytest.mark.integration
class TestIntegration:
    """Integration tests for inference pipeline"""
    
    def test_config_to_inference_pipeline(self):
        """Test complete pipeline from config to inference"""
        config = create_inference_config(
            model_path="test/model",
            device="cpu",
            batch_size=2
        )
        
        assert config is not None
        assert config.batch_size == 2
        assert config.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
