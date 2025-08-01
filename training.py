
from datasets import Dataset, DatasetDict
import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

from peft import LoraConfig
from typing import Dict, Any
import evaluate
import nltk
import torch
import gc
import numpy as np
from trl import SFTConfig, SFTTrainer
nltk.download("punkt")


# NOTE: Open mlflow in colab
import mlflow
import subprocess
from pyngrok import ngrok, conf
import getpass


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI, "--port", "5000"])
mlflow.set_tracking_uri("http://localhost:5000")


NGROK_AUTH_TOKEN = getpass.getpass('Enter your ngrok authtoken: ')
conf.get_default().auth_token = NGROK_AUTH_TOKEN
public_url = ngrok.connect(5000)
print(f"MLflow UI is available at: {public_url}")


# NOTE: Training script



HF_KEY = ''
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
})
print(dataset)

system_message = "You are an expert in extracting tags from Arabic news articles. Your task is to read the provided article text and extract the most relevant keywords that represent the main topics of the article. Return a list of the extracted keywords."

def format_dataset(example, system_message):
  prompt = f"<Article Body>\n{example['body']}<\Article Body>"
  converted_sample = {
       "prompt": [
                    {'role':'system','content':system_message},
                    {"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": example["tags"]}]}
  return converted_sample


formatted_dataset = dataset.map(format_dataset,fn_kwargs={'system_message':system_message},  remove_columns=dataset["train"].column_names)


login(token=HF_KEY)
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)



rank_dimension = 2
lora_alpha = 1
lora_dropout = 0.05

lora_config = LoraConfig(
    r=rank_dimension,  
    lora_alpha=lora_alpha,  
    lora_dropout=lora_dropout,  
    bias="none",  
    target_modules="all-linear",  
    task_type="CAUSAL_LM",  
)

peft_model = get_peft_model(base_model, lora_config)



# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
def compute_metrics(pred):
  pred_ids = pred.predictions
  labels_ids = pred.label_ids
  pred_ids[labels_ids == -100] = tokenizer.pad_token_id
  labels_ids[labels_ids == -100] = tokenizer.pad_token_id
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  rouge_output = rouge_metric.compute(
      predictions=pred_str,
      references=label_str,
      rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
  )
  bleu_output = bleu_metric.compute(
      predictions=pred_str,
      references=label_str,
  )
  return {
      "R1": round(rouge_output["rouge1"], 4),
      "R2": round(rouge_output["rouge2"], 4),
      "RL": round(rouge_output["rougeL"], 4),
      "RLsum": round(rouge_output["rougeLsum"], 4),
      "BLEU": round(bleu_output["bleu"], 4),
  }

def preprocess_logits_for_metrics(logits, labels):
  pred_ids = torch.argmax(logits, dim=-1)
  return pred_ids





# Configure trainer
args = SFTConfig(
    output_dir="./sft_output",
    max_steps=3000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=250,
    save_steps=250,
    eval_strategy="steps",
    eval_steps=250,
    fp16=True,
    report_to="mlflow",

)


mlflow.set_experiment("Arabic News Tag Extraction")

# Start new run
with mlflow.start_run(run_name="peft_lora_training", nested=True) as run:
  print(f"Starting new run with ID: {run.info.run_id}")
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  trainer = SFTTrainer(
      model=peft_model,
      args=args,
      train_dataset=formatted_dataset["train"],
      eval_dataset=formatted_dataset["val"],
      peft_config=lora_config,  
      processing_class=tokenizer,
      compute_metrics=compute_metrics,
      preprocess_logits_for_metrics=preprocess_logits_for_metrics,


  )
  train_result = trainer.train()

  adapter_save_path = "sft_output/lora_adapter_only"
  peft_model.save_pretrained(adapter_save_path)
    # Log to MLflow
  mlflow.log_artifacts(adapter_save_path, artifact_path="lora_adapter")
  model_info = mlflow.transformers.log_model(
      transformers_model={
          "model": peft_model,
          "tokenizer": tokenizer,
      },
      name="peft_model",
  )



# Evalute test set
def preprocess_test_function(example):
    # Ensure both fields exist and are strings
    prompt = str(example.get("prompt", ""))
    completion = str(example.get("completion", ""))

    full_input = prompt + completion

    tokenized = tokenizer(
        full_input,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None  # Use None for Hugging Face datasets
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_test = formatted_dataset["test"].map(
    preprocess_test_function,
    batched=False,
    remove_columns=["prompt", "completion"]
)


test_results = trainer.evaluate(tokenized_test)
print(test_results)