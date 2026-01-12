import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
import gc
from peft import LoraConfig
from trl import SFTTrainer
from src.config import Config
from src.data_loader import load_or_generate_data

def train_qlora_model():
    print("🚀 Starting Fine-Tuning Pipeline...")
    
    # 1. Load Data
    df = load_or_generate_data()
    
    # 2. Enhanced Data Generation (5 Strategies)
    print("   🎨 Generating Diverse Instruction Pairs...")
    instruct_data = []
    
    for _, row in df.iterrows():
        name = row.get('product_name', 'Unknown Product')
        desc = row.get('about_product', 'No description available.')
        cat = row.get('category', 'General')
        price = row.get('discounted_price', 'N/A')
        rating = row.get('rating', 0)
        reviews = str(row.get('review_content', ''))[:300] # truncate reviews
        
        # Strategy A: Creative Writing (Marketing Pitch)
        instruct_data.append({
            "text": f"<s>[INST] Create a compelling marketing pitch for {name}. [/INST] {desc} </s>"
        })
        
        # Strategy B: Logical Reasoning (Price Justification)
        instruct_data.append({
            "text": f"<s>[INST] Why is {name} priced at {price}? [/INST] This item belongs to the {cat} category. With a customer rating of {rating}/5, the price reflects its market positioning and features. </s>"
        })

        # Strategy C: Sentiment Analysis (Summarization)
        if len(reviews) > 10:
            instruct_data.append({
                "text": f"<s>[INST] Summarize the customer sentiment for {name}. [/INST] Users have rated this product {rating}/5. Key feedback includes: {reviews}... </s>"
            })

        # Strategy D: Categorization (Classification)
        instruct_data.append({
            "text": f"<s>[INST] Classify the following product: {name} [/INST] Category: {cat} </s>"
        })

        # Strategy E: Recommendation Engine (Boolean Logic)
        recommendation = "highly recommended" if float(rating) >= 4.0 else "average"
        instruct_data.append({
            "text": f"<s>[INST] Is {name} a good buy? [/INST] Based on a rating of {rating}, this product is considered {recommendation}. </s>"
        })

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(instruct_data)
    print(f"   ✅ Created {len(instruct_data)} training examples.")

    # 3. Load Tokenizer & Fix Padding (CRITICAL FIX)
    print("   ⚙️ Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID, trust_remote_code=True, use_fast=True)
    
    # --- THE FIX ---
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    # ---------------

    # 4. Load Model (4-bit Quantization)
    print("   🤖 Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Target more modules for better learning
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.ADAPTER_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Keep low for GPU memory safety
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=10000, # Short run for demo (remove or increase for full training)
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    # 7. Trainer
    print("   🏋️ Starting SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    
    print("   💾 Saving Adapter...")
    trainer.model.save_pretrained(Config.ADAPTER_PATH)
    print("✅ Fine-Tuning Complete.")
    print("   🧹 Cleaning up Training Memory...")
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    # -------------------------------
    
    print("✅ Fine-Tuning Complete.")

if __name__ == "__main__":
    train_qlora_model()