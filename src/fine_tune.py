import os
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from src.config import Config
from src.data_loader import load_or_generate_data

def generate_reasoning_dataset(df):
    """
    Generates training data that teaches LOGIC, not just memorization.
    """
    data = []
    for _, row in df.iterrows():
        name = row.get('product_name', 'Product')
        cat = row.get('category', 'General')
        price = row.get('discounted_price', 0)
        rating = row.get('rating', 0)
        desc = str(row.get('about_product', ''))[:500]
        
        # 1. The "Analyst" Task (Why is this good?)
        # Teaches the model to connect features to benefits
        data.append({
            "text": (
                f"<s>[INST] Analyze the market appeal of: {name} [/INST] "
                f"This product targets the {cat} segment. "
                f"With a rating of {rating}, it stands out due to features like: {desc}. "
                f"At a price point of {price}, it offers strong value. </s>"
            )
        })

        # 2. The "Comparison" Task (Implicit)
        # Teaches the model to judge quality based on data
        verdict = "Premium choice" if float(rating) > 4.3 else "Budget-friendly option"
        data.append({
            "text": (
                f"<s>[INST] Evaluate the quality-to-price ratio for {name}. [/INST] "
                f"Rating: {rating}/5. Price: {price}. Verdict: {verdict}. "
                f"Reasoning: High user satisfaction suggests the features mentioned ({desc[:50]}...) work as advertised. </s>"
            )
        })
        
    return Dataset.from_list(data)

def train_qlora_model():
    print("🚀 Starting Grandmaster Fine-Tuning...")
    
    # 1. Load & Prep Data
    df = load_or_generate_data()
    dataset = generate_reasoning_dataset(df)
    print(f"   ✅ Generated {len(dataset)} reasoning examples.")

    # 2. Tokenizer (Fix Padding Issue)
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fixed for fp16 training

    # 3. Model (Optimized 4-bit)
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
        trust_remote_code=True,
    )
    
    # Critical for LoRA stability
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    # 4. LoRA Config (The Warning Fix)
    peft_config = LoraConfig(
        lora_alpha=32, # Higher alpha = stronger adaptation
        lora_dropout=0.05,
        r=16, # Rank 16 is better for reasoning than 8
        bias="none",
        task_type=TaskType.CAUSAL_LM, # Explicitly set task type
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target ALL linear layers
    )

    # 5. Training Args (Speed & Stability)
    training_args = TrainingArguments(
        output_dir=Config.ADAPTER_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=4, # Increased slightly for stability
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine", # Cosine decay is better for convergence
        group_by_length=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024, # Increased context
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("   🏋️ Training...")
    trainer.train()
    
    print("   💾 Saving...")
    trainer.model.save_pretrained(Config.ADAPTER_PATH)
    
    # Cleanup
    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Fine-Tuning Complete.")