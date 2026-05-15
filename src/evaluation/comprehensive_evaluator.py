"""
Comprehensive Evaluation Framework for LLM Fine-Tuning
Supports multiple metrics, benchmarks, and production-grade evaluation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework"""
    model_path: str = "./outputs/merged_model"
    device: str = "cuda"
    batch_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Metrics to compute
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_bertscore: bool = True
    compute_custom_metrics: bool = True
    
    # Benchmarking
    run_benchmarks: bool = True
    benchmark_iterations: int = 10


class ComprehensiveEvaluator:
    """
    Production-grade evaluation framework for fine-tuned LLMs
    Supports multiple metrics and comprehensive benchmarking
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logger
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _load_model(self):
        """Load model and tokenizer for evaluation"""
        self.logger.info(f"Loading model from {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device
        )
        
        self.logger.info("Model loaded successfully")
    
    def _initialize_metrics(self):
        """Initialize evaluation metrics"""
        self.logger.info("Initializing evaluation metrics")
        
        self.metrics = {}
        
        if self.config.compute_bleu:
            try:
                self.metrics['bleu'] = load_metric('bleu')
                self.logger.info("BLEU metric initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load BLEU metric: {e}")
        
        if self.config.compute_rouge:
            try:
                self.metrics['rouge'] = load_metric('rouge')
                self.logger.info("ROUGE metric initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load ROUGE metric: {e}")
        
        if self.config.compute_bertscore:
            try:
                self.metrics['bertscore'] = load_metric('bertscore')
                self.logger.info("BERTScore metric initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load BERTScore metric: {e}")
    
    def compute_perplexity(self, dataset) -> Dict[str, float]:
        """
        Compute perplexity on given dataset
        Measures how well the model predicts the data
        """
        self.logger.info("Computing perplexity...")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataset:
                inputs = self.tokenizer(
                    batch['text'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.config.device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].size(0)
                total_tokens += inputs['input_ids'].numel()
        
        perplexity = np.exp(total_loss / total_tokens)
        
        return {
            'perplexity': perplexity,
            'total_loss': total_loss,
            'total_tokens': total_tokens
        }
    
    def compute_bleu_score(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU score for translation/generation quality
        """
        if 'bleu' not in self.metrics:
            return {'bleu': None, 'error': 'BLEU metric not available'}
        
        self.logger.info("Computing BLEU score...")
        
        # Tokenize predictions and references
        tokenized_predictions = [pred.split() for pred in predictions]
        tokenized_references = [[ref.split() for ref in refs] for refs in references]
        
        result = self.metrics['bleu'].compute(
            predictions=tokenized_predictions,
            references=tokenized_references
        )
        
        return {
            'bleu': result['bleu'],
            'precisions': result['precisions'],
            'brevity_penalty': result['brevity_penalty'],
            'length_ratio': result['length_ratio']
        }
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for summarization quality
        """
        if 'rouge' not in self.metrics:
            return {'rouge': None, 'error': 'ROUGE metric not available'}
        
        self.logger.info("Computing ROUGE scores...")
        
        result = self.metrics['rouge'].compute(
            predictions=predictions,
            references=references
        )
        
        return {
            'rouge1': result['rouge1'].mid.fmeasure,
            'rouge2': result['rouge2'].mid.fmeasure,
            'rougeL': result['rougeL'].mid.fmeasure,
            'rougeLsum': result['rougeLsum'].mid.fmeasure
        }
    
    def compute_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity
        """
        if 'bertscore' not in self.metrics:
            return {'bertscore': None, 'error': 'BERTScore metric not available'}
        
        self.logger.info("Computing BERTScore...")
        
        result = self.metrics['bertscore'].compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        
        return {
            'bertscore_precision': np.mean(result['precision']),
            'bertscore_recall': np.mean(result['recall']),
            'bertscore_f1': np.mean(result['f1'])
        }
    
    def compute_custom_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute custom domain-specific metrics
        """
        self.logger.info("Computing custom metrics...")
        
        metrics = {}
        
        # Length-based metrics
        metrics['avg_prediction_length'] = np.mean([len(pred.split()) for pred in predictions])
        metrics['avg_reference_length'] = np.mean([len(ref.split()) for ref in references])
        metrics['length_ratio'] = metrics['avg_prediction_length'] / metrics['avg_reference_length']
        
        # Diversity metrics (unique n-grams)
        def calculate_diversity(texts, n=2):
            all_ngrams = []
            for text in texts:
                words = text.split()
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
            return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0
        
        metrics['bigram_diversity'] = calculate_diversity(predictions, n=2)
        metrics['trigram_diversity'] = calculate_diversity(predictions, n=3)
        
        # Token overlap metrics
        def calculate_overlap(pred, ref):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                return 0
            return len(pred_tokens & ref_tokens) / len(ref_tokens)
        
        overlaps = [calculate_overlap(p, r) for p, r in zip(predictions, references)]
        metrics['avg_token_overlap'] = np.mean(overlaps)
        
        return metrics
    
    def run_inference_benchmark(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Run inference benchmarking to measure performance
        """
        self.logger.info(f"Running inference benchmark with {len(prompts)} prompts...")
        
        self.model.eval()
        results = []
        
        for prompt in prompts:
            start_time = time.time()
            
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'generation_time': generation_time,
                'output_length': len(outputs[0])
            })
        
        # Calculate statistics
        generation_times = [r['generation_time'] for r in results]
        output_lengths = [r['output_length'] for r in results]
        
        return {
            'avg_generation_time': np.mean(generation_times),
            'min_generation_time': np.min(generation_times),
            'max_generation_time': np.max(generation_times),
            'std_generation_time': np.std(generation_times),
            'avg_output_length': np.mean(output_lengths),
            'tokens_per_second': np.mean(output_lengths) / np.mean(generation_times),
            'detailed_results': results
        }
    
    def evaluate_comprehensive(self, 
                            test_dataset,
                            predictions: List[str],
                            references: List[str],
                            prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all configured metrics
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        evaluation_results = {
            'timestamp': time.time(),
            'config': self.config.__dict__
        }
        
        # Compute perplexity if configured
        if self.config.compute_perplexity:
            try:
                perplexity_results = self.compute_perplexity(test_dataset)
                evaluation_results['perplexity'] = perplexity_results
            except Exception as e:
                self.logger.error(f"Error computing perplexity: {e}")
                evaluation_results['perplexity'] = {'error': str(e)}
        
        # Compute BLEU if configured
        if self.config.compute_bleu:
            try:
                bleu_results = self.compute_bleu_score(predictions, [references])
                evaluation_results['bleu'] = bleu_results
            except Exception as e:
                self.logger.error(f"Error computing BLEU: {e}")
                evaluation_results['bleu'] = {'error': str(e)}
        
        # Compute ROUGE if configured
        if self.config.compute_rouge:
            try:
                rouge_results = self.compute_rouge_scores(predictions, references)
                evaluation_results['rouge'] = rouge_results
            except Exception as e:
                self.logger.error(f"Error computing ROUGE: {e}")
                evaluation_results['rouge'] = {'error': str(e)}
        
        # Compute BERTScore if configured
        if self.config.compute_bertscore:
            try:
                bertscore_results = self.compute_bert_score(predictions, references)
                evaluation_results['bertscore'] = bertscore_results
            except Exception as e:
                self.logger.error(f"Error computing BERTScore: {e}")
                evaluation_results['bertscore'] = {'error': str(e)}
        
        # Compute custom metrics if configured
        if self.config.compute_custom_metrics:
            try:
                custom_results = self.compute_custom_metrics(predictions, references)
                evaluation_results['custom_metrics'] = custom_results
            except Exception as e:
                self.logger.error(f"Error computing custom metrics: {e}")
                evaluation_results['custom_metrics'] = {'error': str(e)}
        
        # Run benchmarks if configured
        if self.config.run_benchmarks and prompts:
            try:
                benchmark_results = self.run_inference_benchmark(prompts)
                evaluation_results['benchmarks'] = benchmark_results
            except Exception as e:
                self.logger.error(f"Error running benchmarks: {e}")
                evaluation_results['benchmarks'] = {'error': str(e)}
        
        self.logger.info("Comprehensive evaluation completed")
        
        return evaluation_results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {evaluation_results.get('timestamp', 'N/A')}")
        report.append("")
        
        # Perplexity
        if 'perplexity' in evaluation_results and 'perplexity' in evaluation_results['perplexity']:
            report.append("PERPLEXITY:")
            report.append(f"  Perplexity: {evaluation_results['perplexity']['perplexity']:.2f}")
            report.append("")
        
        # BLEU
        if 'bleu' in evaluation_results and 'bleu' in evaluation_results['bleu']:
            report.append("BLEU SCORE:")
            report.append(f"  BLEU: {evaluation_results['bleu']['bleu']:.4f}")
            report.append(f"  Brevity Penalty: {evaluation_results['bleu']['brevity_penalty']:.4f}")
            report.append("")
        
        # ROUGE
        if 'rouge' in evaluation_results:
            report.append("ROUGE SCORES:")
            report.append(f"  ROUGE-1: {evaluation_results['rouge']['rouge1']:.4f}")
            report.append(f"  ROUGE-2: {evaluation_results['rouge']['rouge2']:.4f}")
            report.append(f"  ROUGE-L: {evaluation_results['rouge']['rougeL']:.4f}")
            report.append("")
        
        # Custom Metrics
        if 'custom_metrics' in evaluation_results:
            report.append("CUSTOM METRICS:")
            for key, value in evaluation_results['custom_metrics'].items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")
        
        # Benchmarks
        if 'benchmarks' in evaluation_results:
            report.append("PERFORMANCE BENCHMARKS:")
            report.append(f"  Avg Generation Time: {evaluation_results['benchmarks']['avg_generation_time']:.2f}s")
            report.append(f"  Tokens/Second: {evaluation_results['benchmarks']['tokens_per_second']:.2f}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)


def create_evaluator(
    model_path: str = "./outputs/merged_model",
    device: str = "cuda",
    **kwargs
) -> ComprehensiveEvaluator:
    """
    Factory function to create evaluator with default configuration
    """
    config = EvaluationConfig(
        model_path=model_path,
        device=device,
        **kwargs
    )
    return ComprehensiveEvaluator(config)


# Example usage
if __name__ == "__main__":
    # Create evaluator
    evaluator = create_evaluator(
        model_path="./outputs/merged_model",
        device="cuda"
    )
    
    # Example data
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "What is the difference between AI and machine learning?"
    ]
    
    predictions = [
        "Quantum computing uses quantum bits to perform complex calculations.",
        "AI is the broader field while machine learning is a subset of AI."
    ]
    
    references = [
        "Quantum computing harnesses quantum mechanics to process information in fundamentally new ways.",
        "Artificial intelligence is the general concept of machines performing intelligent tasks, while machine learning specifically involves algorithms that learn from data."
    ]
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_comprehensive(
        test_dataset=None,  # Would be actual dataset
        predictions=predictions,
        references=references,
        prompts=test_prompts
    )
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(results)
    print(report)
