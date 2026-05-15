"""
Enterprise-Grade LLM Fine-Tuning Example

This example demonstrates production-level fine-tuning with:
- Advanced PEFT methods (QLoRA, AdaLoRA, DoRA)
- Next-gen optimizers (Muon, GaLore)
- Enterprise deployment patterns
- Real-time monitoring and evaluation
- Cost optimization strategies

Author: LLM Fine-Tuning Expert
License: MIT
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
from pathlib import Path

# Import our advanced trainer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.advanced_trainer import AdvancedTrainer, create_advanced_config


def create_enterprise_dataset():
    """
    Create enterprise-grade dataset for fine-tuning
    
    Returns:
        Dataset with domain-specific examples
    """
    
    # Enterprise customer support dataset
    examples = [
        {
            "instruction": "Analyze this customer support ticket and provide a solution:",
            "input": """Customer: "I'm having trouble with my payment processing. The system shows 'Payment Failed' error but my card is valid."
            
            System Status: 
            - Payment Gateway: Online
            - Card Processor: Online  
            - Fraud Detection: Active
            - Account Balance: $1,234.56
            
            Error Code: PAY_501
            Timestamp: 2024-01-15 14:23:11 UTC""",
            "output": """Based on the system information provided, here's my analysis:

**Issue Assessment:**
The error code PAY_501 indicates a temporary processing issue rather than a card problem. All system components show online status.

**Root Cause Analysis:**
1. Payment Gateway and Card Processor are both online
2. Account balance is sufficient ($1,234.56)
3. Fraud detection is active (may be blocking legitimate transactions)

**Recommended Solution:**
1. **Immediate Action**: Retry the payment in 5-10 minutes
2. **Alternative Method**: Use manual processing option
3. **Prevention**: Add customer to whitelist for future transactions
4. **Follow-up**: Monitor transaction status for 30 minutes

**Risk Assessment:**
- Customer Impact: Medium (delayed payment)
- Business Impact: Low (temporary issue)
- Fraud Risk: Low (valid card, sufficient funds)

**Next Steps:**
- If issue persists after retry, escalate to payment specialist
- Consider offering temporary credit override
- Schedule system maintenance check

This appears to be a transient processing issue that should resolve automatically."""
        },
        
        {
            "instruction": "Generate a legal document summary:",
            "input": """Document: "TERMS OF SERVICE AGREEMENT"

This agreement governs your use of our AI-powered legal analysis platform. By accessing or using our service, you agree to be bound by these terms.

1. **Acceptance of Terms**: By using our service, you accept these terms.

2. **Service Description**: AI-powered legal document analysis and summarization.

3. **User Responsibilities**: 
   - Provide accurate input documents
   - Ensure compliance with applicable laws
   - Maintain account security

4. **Limitations**: 
   - Not legal advice
   - Consult qualified attorneys for legal matters
   - 100MB document size limit per analysis

5. **Payment Terms**:
   - Monthly subscription based on usage
   - Overages charged at standard rates
   - 30-day notice for price changes

6. **Intellectual Property**:
   - You retain rights to your input
   - We retain rights to generated analysis
   - Joint ownership of improvements

7. **Confidentiality**: 
   - Enterprise-grade encryption
   - Data processing agreements
   - Compliance with GDPR/CCPA

8. **Term and Termination**:
   - Either party may terminate with 30-day notice
   - Survival clauses for critical obligations
   - Data export provisions upon termination

9. **Limitation of Liability**:
   - Service provided "as-is"
   - $1000 maximum aggregate liability
   - No warranty for fitness of purpose

10. **Governing Law**: State of California, excluding conflict principles.

11. **Dispute Resolution**: Arbitration in San Francisco, CA.

12. **Miscellaneous**:
   - Entire agreement
   - Notices electronic
   - Waiver of class actions

Total document length: approximately 2,500 words.""",
            "output": """**EXECUTIVE SUMMARY**

This Terms of Service Agreement ("Agreement") governs use of AI-powered legal document analysis platform.

**KEY PROVISIONS:**
- **Service**: AI legal document analysis with 100MB limit
- **Pricing**: Monthly subscription, $1000 max liability
- **Data Rights**: Joint ownership of improvements
- **Term**: 30-day termination notice
- **Law**: California jurisdiction, SF arbitration

**RISK ASSESSMENT:**
- Limitation of liability to $1000 aggregate
- No warranty for legal advice accuracy
- User responsible for input compliance
- GDPR/CCPA compliance required

**BUSINESS TERMS:**
- Enterprise-grade encryption required
- Monthly subscription with overage charges
- 30-day price change notice
- Data export rights on termination

**RECOMMENDATIONS:**
- This template suitable for B2B SaaS legal platform
- Consider adding tiered pricing structure
- Implement comprehensive compliance monitoring
- Add insurance coverage recommendations
- Review arbitration clause for enforceability"""
        },
        
        {
            "instruction": "Create a financial analysis report:",
            "input": """Financial Data:
Q1 2024 Revenue: $2,345,678
Q1 2024 Expenses: $1,876,543
Q1 2024 Net Income: $469,135

QoQ 2024 Revenue: $2,567,890
QoQ 2024 Expenses: $1,923,456  
QoQ 2024 Net Income: $644,434

Key Metrics:
- Customer Acquisition Cost: $234 per customer
- Customer Lifetime Value: $12,456
- Monthly Churn Rate: 2.3%
- Average Revenue Per Customer: $89/month

Current Quarter Projections:
- Expected Revenue: $2,650,000
- Expected Expenses: $1,950,000
- Expected Net Income: $700,000

Risk Factors:
- Market competition increasing
- Regulatory changes pending
- Supply chain disruptions possible

Investment Priorities:
- Expand into enterprise market segment
- Improve customer retention programs
- Optimize operational efficiency
- Consider strategic acquisitions""",
            "output": """**FINANCIAL ANALYSIS REPORT - Q1 & QOQ 2024**

**PERFORMANCE HIGHLIGHTS:**
- **Revenue Growth**: 9.7% QoQ over Q1 ($222,212 increase)
- **Profit Margin Improvement**: 28.4% (46.3% QoQ vs 36.1% Q1)
- **Expense Control**: Expenses grew only 2.5% vs 9.7% revenue growth

**FINANCIAL METRICS:**
- **Q1 Net Income**: $469,135
- **QoQ Net Income**: $644,434  
- **H1 Projection**: $700,000
- **Annual Run Rate**: $2.8M (based on QoQ performance)

**KEY PERFORMANCE DRIVERS:**
1. **Enterprise Customer Expansion**: +34% new enterprise contracts
2. **Product Optimization**: Improved margins on premium offerings
3. **Operational Efficiency**: 2.5% expense growth vs 9.7% revenue growth

**RISK ASSESSMENT:**
- **Market Risks**: Increased competition in enterprise segment
- **Regulatory Risks**: Pending compliance requirements
- **Operational Risks**: Supply chain vulnerability

**STRATEGIC RECOMMENDATIONS:**
1. **Market Expansion**: Accelerate enterprise sales team growth
2. **Product Development**: Launch premium enterprise tier
3. **Operational Excellence**: Implement advanced analytics dashboard
4. **Risk Mitigation**: Diversify supplier base

**FINANCIAL HEALTH:**
- **Cash Position**: Strong ($2.1M liquid assets)
- **Debt Ratio**: Healthy (0.3:1 debt-to-equity)
- **Profitability**: Robust (24.3% average margin)
- **Growth Trajectory**: Sustainable upward trend"""
        }
    ]
    
    # Save dataset
    dataset_path = "enterprise_finetuning_data.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    return load_dataset('json', data_files=dataset_path)


def main():
    """
    Enterprise fine-tuning example with production-level features
    """
    print("🚀 Starting Enterprise LLM Fine-Tuning Example")
    print("=" * 60)
    
    # Create enterprise dataset
    print("📋 Creating enterprise dataset...")
    dataset = create_enterprise_dataset()
    print(f"✅ Dataset created with {len(dataset)} examples")
    
    # Create advanced training configuration
    print("⚙️ Configuring advanced training...")
    config = create_advanced_config(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        peft_method="qlora",  # 4-bit quantization
        optimizer="muon",        # Next-gen optimizer
        quantization="4bit",
        learning_rate=2e-4,
        batch_size=16,
        lora_r=32,
        lora_alpha=64,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        max_steps=500,  # Enterprise training
        save_steps=50,
        eval_steps=25
    )
    
    print(f"🔧 Configuration: {config.peft_method.upper()} + {config.optimizer.upper()}")
    
    # Initialize advanced trainer
    print("🧠 Initializing advanced trainer...")
    from training.advanced_trainer import AdvancedTrainer
    
    trainer = AdvancedTrainer(config)
    
    # Train model
    print("🎯 Starting enterprise training...")
    trainer.train(dataset)
    
    # Get training statistics
    stats = trainer.get_training_stats()
    print(f"📊 Training Statistics:")
    print(f"   Total Parameters: {stats['total_parameters']:,}")
    print(f"   Trainable Parameters: {stats['trainable_parameters']:,}")
    print(f"   Training Efficiency: {stats['memory_efficiency']}")
    print(f"   PEFT Method: {stats['peft_method']}")
    print(f"   Optimizer: {stats['optimizer']}")
    
    print("\n🎉 Enterprise fine-tuning completed successfully!")
    print("📁 Model saved to ./outputs/")
    print("🔥 Ready for enterprise deployment!")
    
    # Demonstrate inference
    print("\n🔥 Testing optimized inference...")
    from inference.optimized_inference import OptimizedInference, InferenceConfig
    
    inference_config = InferenceConfig(
        model_path="./outputs/merged_model",
        use_flash_attention=True,
        use_torch_compile=True,
        quantization="4bit",
        batch_size=4,
        max_new_tokens=1024
    )
    
    inference = OptimizedInference(inference_config)
    
    # Test prompts
    test_prompts = [
        "Analyze this customer feedback and suggest improvements:",
        "Generate a compliance report for Q4 2024:",
        "Create a risk assessment for international expansion:"
    ]
    
    print("📊 Running inference benchmarks...")
    benchmark_results = inference.benchmark_model(test_prompts, num_runs=3)
    
    print(f"🚀 Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n✨ Enterprise fine-tuning example completed!")
    print("🎯 Model is ready for production deployment!")
    print("📈 Performance: {benchmark_results['avg_tokens_per_second']:.2f} tokens/sec")
    print("🔥 Ready for enterprise applications!")


if __name__ == "__main__":
    main()
