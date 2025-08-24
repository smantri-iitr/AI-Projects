# Dialogue Summarization with QLoRA Fine-tuned Llama-2
# This implementation includes data preparation, QLoRA fine-tuning, training, and ROUGE evaluation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset as HFDataset
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the model and training"""
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Base Llama-2 model
    max_length: int = 512
    max_target_length: int = 128
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4
    
    # QLoRA specific parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"]

class DialogueSummarizationDataset(Dataset):
    """Custom dataset for dialogue summarization"""
    
    def __init__(self, dialogues: List[str], summaries: List[str], 
                 tokenizer, max_length: int = 512, max_target_length: int = 128):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
        # Add special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        
        # Format the input with instruction
        prompt = f"### Instruction:\nSummarize the following dialogue:\n\n### Dialogue:\n{dialogue}\n\n### Summary:\n"
        full_text = prompt + summary + self.tokenizer.eos_token
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids, but with -100 for prompt tokens)
        labels = encoding["input_ids"].clone()
        
        # Find where the summary starts
        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)
        prompt_length = len(prompt_encoding["input_ids"])
        
        # Set prompt tokens to -100 (ignored in loss calculation)
        labels[:, :prompt_length] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

class DialogueSummarizer:
    """Main class for dialogue summarization with QLoRA fine-tuning"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                   use_stemmer=True)
        
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer with QLoRA configuration"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
    def prepare_data(self, train_dialogues: List[str], train_summaries: List[str],
                    val_dialogues: List[str] = None, val_summaries: List[str] = None):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets...")
        
        # Create training dataset
        train_dataset = DialogueSummarizationDataset(
            train_dialogues, train_summaries, self.tokenizer,
            self.config.max_length, self.config.max_target_length
        )
        
        # Create validation dataset if provided
        val_dataset = None
        if val_dialogues and val_summaries:
            val_dataset = DialogueSummarizationDataset(
                val_dialogues, val_summaries, self.tokenizer,
                self.config.max_length, self.config.max_target_length
            )
            
        return train_dataset, val_dataset
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores for evaluation"""
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"].append(scores['rouge1'].fmeasure)
            rouge_scores["rouge2"].append(scores['rouge2'].fmeasure)
            rouge_scores["rougeL"].append(scores['rougeL'].fmeasure)
        
        # Calculate averages
        avg_scores = {
            "rouge1": np.mean(rouge_scores["rouge1"]),
            "rouge2": np.mean(rouge_scores["rouge2"]),
            "rougeL": np.mean(rouge_scores["rougeL"])
        }
        
        return avg_scores
    
    def generate_summary(self, dialogue: str, max_new_tokens: int = 128) -> str:
        """Generate summary for a single dialogue"""
        prompt = f"### Instruction:\nSummarize the following dialogue:\n\n### Dialogue:\n{dialogue}\n\n### Summary:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=self.config.max_length)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated summary
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_response.split("### Summary:\n")[-1].strip()
        
        return summary
    
    def train(self, train_dataset, val_dataset=None, output_dir="./dialogue_summarizer_qlora"):
        """Train the model using QLoRA"""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            bf16=True,
            tf32=True,
            report_to=None  # Disable wandb/tensorboard logging
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
    
    def evaluate(self, test_dialogues: List[str], test_summaries: List[str]) -> Dict[str, float]:
        """Evaluate the model using ROUGE scores"""
        logger.info("Evaluating model...")
        
        predictions = []
        for dialogue in test_dialogues:
            summary = self.generate_summary(dialogue)
            predictions.append(summary)
        
        # Compute ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, test_summaries)
        
        logger.info("ROUGE Scores:")
        for metric, score in rouge_scores.items():
            logger.info(f"{metric.upper()}: {score:.4f}")
        
        return rouge_scores, predictions

# Example usage and helper functions
def load_sample_data():
    """Load or create sample dialogue-summary pairs"""
    # Sample data - replace with your actual dataset
    sample_data = [
        {
            "dialogue": "Person A: Hi, how are you doing today?\nPerson B: I'm doing well, thanks for asking. How about you?\nPerson A: I'm good too. I wanted to talk about the project deadline.\nPerson B: Sure, what about it?\nPerson A: I think we might need an extension. There are some complications.\nPerson B: What kind of complications?\nPerson A: The data we received is incomplete, and we need more time to gather additional information.\nPerson B: I understand. Let me talk to the manager about extending the deadline.",
            "summary": "Person A discusses project deadline concerns with Person B due to incomplete data, and Person B agrees to speak with the manager about an extension."
        },
        {
            "dialogue": "Customer: Hello, I have a problem with my recent order.\nSupport: Hi! I'm sorry to hear that. Can you tell me your order number?\nCustomer: It's OR-12345. I received the wrong item.\nSupport: Let me check that for you. What did you receive instead?\nCustomer: I ordered a blue shirt size M, but got a red shirt size L.\nSupport: I apologize for the mix-up. We'll send you the correct item right away and provide a return label for the wrong item.\nCustomer: Thank you, that would be great. When can I expect the replacement?\nSupport: It should arrive within 3-5 business days.",
            "summary": "Customer received wrong item (red shirt L instead of blue shirt M) for order OR-12345. Support arranged replacement delivery in 3-5 days with return label."
        }
    ]
    
    dialogues = [item["dialogue"] for item in sample_data]
    summaries = [item["summary"] for item in sample_data]
    
    return dialogues, summaries

def main():
    """Main function to demonstrate the dialogue summarization pipeline"""
    
    # Configuration
    config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",  # Make sure you have access to this model
        batch_size=2,  # Reduce for memory constraints
        num_epochs=1,   # Reduce for quick testing
        max_length=512,
        max_target_length=128
    )
    
    # Initialize summarizer
    summarizer = DialogueSummarizer(config)
    
    # Setup model and tokenizer
    summarizer.setup_model_and_tokenizer()
    
    # Load sample data (replace with your actual data loading)
    dialogues, summaries = load_sample_data()
    
    # Split data (for demonstration, using same data for train/val/test)
    train_dialogues, train_summaries = dialogues, summaries
    val_dialogues, val_summaries = dialogues, summaries
    test_dialogues, test_summaries = dialogues, summaries
    
    # Prepare datasets
    train_dataset, val_dataset = summarizer.prepare_data(
        train_dialogues, train_summaries, 
        val_dialogues, val_summaries
    )
    
    # Train the model
    summarizer.train(train_dataset, val_dataset)
    
    # Evaluate the model
    rouge_scores, predictions = summarizer.evaluate(test_dialogues, test_summaries)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for i, (dialogue, reference, prediction) in enumerate(zip(test_dialogues, test_summaries, predictions)):
        print(f"\nExample {i+1}:")
        print(f"Reference: {reference}")
        print(f"Prediction: {prediction}")
        print("-" * 30)
    
    print(f"\nOverall ROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    # Note: Make sure you have the required dependencies installed:
    # pip install torch transformers peft datasets accelerate bitsandbytes rouge-score
    main()