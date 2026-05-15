"""
Real-World Dataset Examples and Pipeline for LLM Fine-Tuning
Demonstrates production-grade data processing for various use cases
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import pandas as pd
import json

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    dataset_name: str
    split: str = "train"
    max_samples: Optional[int] = None
    text_field: str = "text"
    instruction_field: Optional[str] = None
    response_field: Optional[str] = None
    context_field: Optional[str] = None


class RealWorldDatasetProcessor:
    """
    Production-grade dataset processor for real-world fine-tuning scenarios
    Supports multiple dataset formats and preprocessing pipelines
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logger
        
    def load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace or local source"""
        self.logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            # Try loading from HuggingFace
            dataset = load_dataset(self.config.dataset_name, split=self.config.split)
            
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            
            self.logger.info(f"Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_instruction_tuning(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for instruction tuning
        Format: Instruction: {instruction}\nContext: {context}\nResponse: {response}
        """
        self.logger.info("Preprocessing for instruction tuning")
        
        def format_instruction(example):
            if self.config.instruction_field and self.config.response_field:
                instruction = example[self.config.instruction_field]
                response = example[self.config.response_field]
                
                if self.config.context_field and example.get(self.config.context_field):
                    context = example[self.config.context_field]
                    text = f"Instruction: {instruction}\nContext: {context}\nResponse: {response}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {response}"
                
                return {'text': text}
            else:
                return example
        
        processed_dataset = dataset.map(format_instruction)
        return processed_dataset
    
    def preprocess_continuation(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for text continuation
        """
        self.logger.info("Preprocessing for text continuation")
        
        def format_continuation(example):
            text = example[self.config.text_field]
            return {'text': text}
        
        processed_dataset = dataset.map(format_continuation)
        return processed_dataset
    
    def preprocess_chat_format(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for chat format fine-tuning
        Format suitable for conversational models
        """
        self.logger.info("Preprocessing for chat format")
        
        def format_chat(example):
            if self.config.instruction_field and self.config.response_field:
                messages = [
                    {"role": "user", "content": example[self.config.instruction_field]},
                    {"role": "assistant", "content": example[self.config.response_field]}
                ]
                return {'messages': messages}
            return example
        
        processed_dataset = dataset.map(format_chat)
        return processed_dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8) -> Dict[str, Dataset]:
        """Split dataset into train and validation sets"""
        self.logger.info(f"Splitting dataset with train ratio: {train_ratio}")
        
        train_size = int(len(dataset) * train_ratio)
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        return {
            'train': train_dataset,
            'eval': eval_dataset
        }
    
    def save_processed_dataset(self, dataset: Dataset, output_path: str):
        """Save processed dataset to disk"""
        self.logger.info(f"Saving processed dataset to {output_path}")
        dataset.save_to_disk(output_path)


# Real-world dataset examples
class EnterpriseDatasetExamples:
    """Examples of real-world enterprise datasets for fine-tuning"""
    
    @staticmethod
    def customer_support_dataset() -> DatasetConfig:
        """Customer support conversation dataset"""
        return DatasetConfig(
            dataset_name="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            text_field="instruction",
            instruction_field="instruction",
            response_field="response",
            max_samples=1000
        )
    
    @staticmethod
    def technical_documentation_dataset() -> DatasetConfig:
        """Technical documentation Q&A dataset"""
        return DatasetConfig(
            dataset_name="tau/stackexchange",
            split="train",
            text_field="text",
            instruction_field="question",
            response_field="answer",
            max_samples=500
        )
    
    @staticmethod
    def code_generation_dataset() -> DatasetConfig:
        """Code generation and explanation dataset"""
        return DatasetConfig(
            dataset_name="codeparrot/github-code",
            split="train",
            text_field="code",
            max_samples=300
        )
    
    @staticmethod
    def medical_qa_dataset() -> DatasetConfig:
        """Medical question answering dataset"""
        return DatasetConfig(
            dataset_name="medalpaca/medical_meadow_medical_flashcards",
            split="train",
            instruction_field="input",
            response_field="output",
            max_samples=200
        )
    
    @staticmethod
    def financial_analysis_dataset() -> DatasetConfig:
        """Financial analysis dataset"""
        return DatasetConfig(
            dataset_name="FinGPT/fingpt-sentiment-cls",
            split="train",
            text_field="text",
            max_samples=400
        )
    
    @staticmethod
    def legal_documentation_dataset() -> DatasetConfig:
        """Legal documentation dataset"""
        return DatasetConfig(
            dataset_name="nguyenkn/constitutional-ai",
            split="train",
            instruction_field="question",
            response_field="answer",
            max_samples=150
        )


# Production pipeline example
class ProductionDataPipeline:
    """
    Production-grade data pipeline for fine-tuning
    Handles data loading, preprocessing, validation, and storage
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processor = RealWorldDatasetProcessor(config)
        self.logger = logger
    
    def run_pipeline(self, 
                    preprocessing_type: str = "instruction_tuning",
                    output_dir: str = "./processed_data") -> Dict[str, Dataset]:
        """
        Run complete data pipeline
        
        Args:
            preprocessing_type: Type of preprocessing (instruction_tuning, continuation, chat_format)
            output_dir: Directory to save processed data
        
        Returns:
            Dictionary with train and eval datasets
        """
        self.logger.info("Starting production data pipeline")
        
        # Load dataset
        dataset = self.processor.load_dataset()
        
        # Preprocess based on type
        if preprocessing_type == "instruction_tuning":
            processed_dataset = self.processor.preprocess_instruction_tuning(dataset)
        elif preprocessing_type == "continuation":
            processed_dataset = self.processor.preprocess_continuation(dataset)
        elif preprocessing_type == "chat_format":
            processed_dataset = self.processor.preprocess_chat_format(dataset)
        else:
            raise ValueError(f"Unknown preprocessing type: {preprocessing_type}")
        
        # Split dataset
        split_datasets = self.processor.split_dataset(processed_dataset)
        
        # Save processed datasets
        self.processor.save_processed_dataset(
            split_datasets['train'],
            f"{output_dir}/train"
        )
        self.processor.save_processed_dataset(
            split_datasets['eval'],
            f"{output_dir}/eval"
        )
        
        self.logger.info(f"Pipeline completed. Train samples: {len(split_datasets['train'])}, "
                        f"Eval samples: {len(split_datasets['eval'])}")
        
        return split_datasets
    
    def validate_data_quality(self, dataset: Dataset) -> Dict[str, any]:
        """
        Validate data quality metrics
        """
        self.logger.info("Validating data quality")
        
        metrics = {
            'total_samples': len(dataset),
            'avg_text_length': 0,
            'min_text_length': float('inf'),
            'max_text_length': 0,
            'empty_samples': 0
        }
        
        text_lengths = []
        for example in dataset:
            if 'text' in example:
                text = example['text']
                length = len(text.split())
                text_lengths.append(length)
                
                if length == 0:
                    metrics['empty_samples'] += 1
            elif 'messages' in example:
                # For chat format
                messages = example['messages']
                total_length = sum(len(msg['content'].split()) for msg in messages)
                text_lengths.append(total_length)
        
        if text_lengths:
            metrics['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            metrics['min_text_length'] = min(text_lengths)
            metrics['max_text_length'] = max(text_lengths)
        
        self.logger.info(f"Data quality metrics: {metrics}")
        return metrics


# Usage examples
def example_customer_support_pipeline():
    """Example: Customer support fine-tuning pipeline"""
    config = EnterpriseDatasetExamples.customer_support_dataset()
    pipeline = ProductionDataPipeline(config)
    
    datasets = pipeline.run_pipeline(
        preprocessing_type="instruction_tuning",
        output_dir="./data/customer_support"
    )
    
    quality_metrics = pipeline.validate_data_quality(datasets['train'])
    
    return datasets, quality_metrics


def example_technical_documentation_pipeline():
    """Example: Technical documentation fine-tuning pipeline"""
    config = EnterpriseDatasetExamples.technical_documentation_dataset()
    pipeline = ProductionDataPipeline(config)
    
    datasets = pipeline.run_pipeline(
        preprocessing_type="instruction_tuning",
        output_dir="./data/technical_docs"
    )
    
    return datasets


def example_code_generation_pipeline():
    """Example: Code generation fine-tuning pipeline"""
    config = EnterpriseDatasetExamples.code_generation_dataset()
    pipeline = ProductionDataPipeline(config)
    
    datasets = pipeline.run_pipeline(
        preprocessing_type="continuation",
        output_dir="./data/code_generation"
    )
    
    return datasets


if __name__ == "__main__":
    import sys
    
    # Run example pipeline
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1]
        
        if dataset_type == "customer_support":
            datasets, metrics = example_customer_support_pipeline()
            print(f"Customer Support Pipeline: Train={len(datasets['train'])}, "
                  f"Eval={len(datasets['eval'])}")
            print(f"Quality Metrics: {metrics}")
        
        elif dataset_type == "technical_docs":
            datasets = example_technical_documentation_pipeline()
            print(f"Technical Docs Pipeline: Train={len(datasets['train'])}, "
                  f"Eval={len(datasets['eval'])}")
        
        elif dataset_type == "code_generation":
            datasets = example_code_generation_pipeline()
            print(f"Code Generation Pipeline: Train={len(datasets['train'])}, "
                  f"Eval={len(datasets['eval'])}")
        
        else:
            print(f"Unknown dataset type: {dataset_type}")
            print("Available: customer_support, technical_docs, code_generation")
    else:
        print("Usage: python real_world_datasets.py <dataset_type>")
        print("Available dataset types:")
        print("  - customer_support")
        print("  - technical_docs")
        print("  - code_generation")
