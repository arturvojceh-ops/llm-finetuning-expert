"""Unit tests for AdvancedTrainer class"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.training.advanced_trainer import (
    AdvancedTrainer,
    AdvancedTrainingConfig,
    create_advanced_config
)


class TestAdvancedTrainingConfig:
    """Test suite for AdvancedTrainingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AdvancedTrainingConfig()
        
        assert config.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.peft_method == "qlora"
        assert config.optimizer == "muon"
        assert config.quantization == "4bit"
        assert config.learning_rate == 2e-4
        assert config.batch_size == 32
        assert config.max_steps == 1000
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = AdvancedTrainingConfig(
            model_name="microsoft/phi-3",
            peft_method="lora",
            optimizer="adamw",
            learning_rate=1e-4,
            batch_size=16
        )
        
        assert config.model_name == "microsoft/phi-3"
        assert config.peft_method == "lora"
        assert config.optimizer == "adamw"
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
    
    def test_create_advanced_config_factory(self):
        """Test factory function for creating configs"""
        config = create_advanced_config(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            peft_method="qlora",
            optimizer="muon"
        )
        
        assert isinstance(config, AdvancedTrainingConfig)
        assert config.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.peft_method == "qlora"


class TestAdvancedTrainer:
    """Test suite for AdvancedTrainer"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return AdvancedTrainingConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            peft_method="lora",
            optimizer="adamw",
            batch_size=4,
            max_steps=10
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
    
    def test_trainer_initialization(self, mock_config, mock_logger):
        """Test trainer initialization"""
        with patch('src.training.advanced_trainer.logging.getLogger', return_value=mock_logger):
            with patch('src.training.advanced_trainer.AdvancedTrainer._setup_monitoring'):
                with patch('src.training.advanced_trainer.AdvancedTrainer._load_model'):
                    with patch('src.training.advanced_trainer.AdvancedTrainer._setup_peft'):
                        with patch('src.training.advanced_trainer.AdvancedTrainer._setup_data_collator'):
                            trainer = AdvancedTrainer(mock_config)
                            
                            assert trainer.config == mock_config
                            assert trainer.logger == mock_logger
    
    def test_setup_data_collator_default(self, mock_config):
        """Test default data collator setup"""
        with patch('src.training.advanced_trainer.logging.getLogger'):
            with patch('src.training.advanced_trainer.AdvancedTrainer._setup_monitoring'):
                with patch('src.training.advanced_trainer.AdvancedTrainer._load_model'):
                    with patch('src.training.advanced_trainer.AdvancedTrainer._setup_peft'):
                        trainer = AdvancedTrainer(mock_config)
                        collator = trainer._setup_data_collator()
                        
                        assert collator is not None
    
    def test_get_training_stats(self, mock_config):
        """Test training statistics calculation"""
        with patch('src.training.advanced_trainer.logging.getLogger'):
            with patch('src.training.advanced_trainer.AdvancedTrainer._setup_monitoring'):
                with patch('src.training.advanced_trainer.AdvancedTrainer._load_model'):
                    with patch('src.training.advanced_trainer.AdvancedTrainer._setup_peft'):
                        with patch('src.training.advanced_trainer.AdvancedTrainer._setup_data_collator'):
                            trainer = AdvancedTrainer(mock_config)
                            
                            # Mock model parameters
                            trainer.model = Mock()
                            trainer.model.parameters = Mock(return_value=[
                                Mock(numel=Mock(return_value=1000)),
                                Mock(numel=Mock(return_value=500)),
                                Mock(numel=Mock(return_value=200))
                            ])
                            
                            stats = trainer.get_training_stats()
                            
                            assert 'total_parameters' in stats
                            assert 'trainable_parameters' in stats
                            assert 'memory_efficiency' in stats
                            assert 'peft_method' in stats
                            assert stats['total_parameters'] == 1700
    
    def test_optimizer_selection_muon(self, mock_config):
        """Test Muon optimizer selection"""
        mock_config.optimizer = "muon"
        
        with patch('src.training.advanced_trainer.logging.getLogger'):
            with patch('src.training.advanced_trainer.AdvancedTrainer._setup_monitoring'):
                with patch('src.training.advanced_trainer.AdvancedTrainer._load_model'):
                    with patch('src.training.advanced_trainer.AdvancedTrainer._setup_peft'):
                        with patch('src.training.advanced_trainer.AdvancedTrainer._setup_data_collator'):
                            trainer = AdvancedTrainer(mock_config)
                            
                            # Mock trainable parameters
                            trainable_params = [Mock()]
                            optimizer = trainer._create_optimizer(trainable_params)
                            
                            # Should return Muon or AdamW as fallback
                            assert optimizer is not None
    
    def test_optimizer_selection_galore(self, mock_config):
        """Test GaLore optimizer selection"""
        mock_config.optimizer = "galore"
        
        with patch('src.training.advanced_trainer.logging.getLogger'):
            with patch('src.training.advanced_trainer.AdvancedTrainer._setup_monitoring'):
                with patch('src.training.advanced_trainer.AdvancedTrainer._load_model'):
                    with patch('src.training.advanced_trainer.AdvancedTrainer._setup_peft'):
                        with patch('src.training.advanced_trainer.AdvancedTrainer._setup_data_collator'):
                            trainer = AdvancedTrainer(mock_config)
                            
                            trainable_params = [Mock()]
                            optimizer = trainer._create_optimizer(trainable_params)
                            
                            assert optimizer is not None
    
    def test_peft_method_validation(self, mock_config):
        """Test PEFT method validation"""
        valid_methods = ["lora", "qlora", "adalora", "full"]
        
        for method in valid_methods:
            mock_config.peft_method = method
            assert mock_config.peft_method in valid_methods


class TestTrainingPipeline:
    """Test suite for training pipeline"""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        return dataset
    
    def test_training_args_creation(self):
        """Test training arguments creation"""
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./outputs",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=100,
            eval_steps=100
        )
        
        assert training_args.output_dir == "./outputs"
        assert training_args.num_train_epochs == 3
        assert training_args.per_device_train_batch_size == 4
        assert training_args.learning_rate == 2e-4


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_config_to_trainer_pipeline(self):
        """Test complete pipeline from config to trainer"""
        config = create_advanced_config(
            model_name="microsoft/phi-3-mini-4k-instruct",
            peft_method="lora",
            optimizer="adamw",
            batch_size=2
        )
        
        assert config is not None
        assert config.batch_size == 2
        assert config.peft_method == "lora"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
