import os
import sys
from typing import Tuple, Union

import fire
import torch
from datasets import load_dataset
from handler import DataHandler
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_absolute_error
import re
import json
from eval import *
from inferer import Inferer  # 导入Inferer类
import numpy as np
from tqdm import tqdm

sampling = {
    "do_sample" : True,
    "top_k": 50, 
    "num_beams": 1,
    "max_new_tokens": 128, 
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9
}

cache_dir = "/root/workspace/cv3ulf4p420c73fli4a0/cache"

def strip_special_chars(text: str) -> str:
    """
    删除特殊字符和多余的空白
    """
    # 移除特殊字符
    text = re.sub(r'[^A-Za-z0-9\s.,-]', '', text)
    # 移除多余空白
    text = ' '.join(text.split())
    return text.strip()

def starts_with_capital_letter(text: str) -> bool:
    """
    检查文本是否以大写字母开头
    """
    if not text:
        return False
    return text[0].isupper()

def extract_sri_value(response: str) -> float:
    """
    从响应中提取SRI值
    
    Args:
        response: 模型的响应文本
    
    Returns:
        float: 提取的SRI值
        
    Raises:
        ValueError: 当无法提取有效的SRI值时
    """
    # 清理文本
    cleaned_response = strip_special_chars(response)
    
    try:
        # 尝试直接转换为float
        return float(cleaned_response)
    except ValueError:
        # 如果直接转换失败，尝试使用正则表达式提取数字
        match = re.search(r'(\d+\.?\d*)', cleaned_response)
        if match:
            return float(match.group(1))
        raise ValueError(f"无法从响应中提取有效的SRI值: {response}")

def evaluate_model(inferer, eval_data, ntries=3, sampling=None):
    """
    评估模型性能
    
    Args:
        inferer: 模型推理器
        eval_data: 评估数据
        ntries: 重试次数
        sampling: 采样策略参数
    
    Returns:
        list: 包含预测结果的列表
    """
    predictions = []
    
    # 使用tqdm显示进度
    pbar = tqdm(eval_data)
    pbar.set_description_str("Evaluating")
    
    for i, item in enumerate(pbar):
        input_text = item["input"]
        ground_truth_sri = item["output"]
        
        # 多次尝试获取有效响应
        for j in range(ntries):
            response = inferer(
                instruction="Based on the following data, output only a single SRI numerical value. Example: 3.5",
                input=input_text,
                **sampling if sampling else {}
            )
            
            # 清理响应文本
            cleaned_response = strip_special_chars(response)
            
            # 尝试提取SRI值
            try:
                sri_value = extract_sri_value(cleaned_response)
                pbar.set_postfix_str("")
                break
            except ValueError:
                pbar.set_postfix_str(f"Invalid output, retrying {j+1}/{ntries}")
                if j == ntries - 1:  # 如果是最后一次尝试
                    sri_value = None
        
        # 记录预测结果
        predictions.append({
            "input": input_text,
            "response": response,
            "cleaned_response": cleaned_response,
            "predicted_sri": sri_value,
            "ground_truth_sri": float(ground_truth_sri)
        })
        
        # 实时显示当前评估指标
        if sri_value is not None:
            current_mse = np.mean([
                (p["predicted_sri"] - p["ground_truth_sri"])**2 
                for p in predictions 
                if p["predicted_sri"] is not None
            ])
            current_mae = np.mean([
                abs(p["predicted_sri"] - p["ground_truth_sri"])
                for p in predictions 
                if p["predicted_sri"] is not None
            ])
            pbar.set_description(f"MSE: {current_mse:.4f}, MAE: {current_mae:.4f}")
    
    return predictions

def calculate_metrics(predictions):
    """
    计算评估指标
    
    Args:
        predictions: 预测结果列表
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    valid_predictions = [p for p in predictions if p["predicted_sri"] is not None]
    
    if not valid_predictions:
        return {
            "mse": float('nan'),
            "mae": float('nan'),
            "valid_ratio": 0.0
        }
    
    mse = np.mean([
        (p["predicted_sri"] - p["ground_truth_sri"])**2 
        for p in valid_predictions
    ])
    
    mae = np.mean([
        abs(p["predicted_sri"] - p["ground_truth_sri"])
        for p in valid_predictions
    ])
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "valid_ratio": len(valid_predictions) / len(predictions)
    }

def main(
    model: str, # e.g. "decapoda-research/llama-7b-hf"
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "medalpaca/prompts/medalpaca.json",
    model_max_length: int = 256,  # should not exceed 2048, as LLaMA is trained with this
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    data_path: str = "medical_meadow_small.json",
    train_in_8bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    per_device_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    global_batch_size: int = 128,
    output_dir: str = "./output",
    save_total_limit: int = 3,
    eval_steps: int = 200,
    device_map: str = "auto",
    group_by_length: bool = False,
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 100,
    fsdp: str = "full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap: str = "LlamaDecoderLayer",
    **kwargs
):
    """
    Trains a large language model using HuggingFace Transformers with custom configuration options.

    Args:
    model (str, optional):
        The model identifier on HuggingFace Model Hub.
    val_set_size (Union[int, float], optional):
        The proportion or number of samples to use for validation. Default is 0.1.
    prompt_template (str, optional):
        The path to the JSON file containing prompt templates. Default is "prompts/medalpaca.json".
    model_max_length (int, optional):
        The maximum length for model inputs. Default is 256.
    train_on_inputs (bool, optional):
        Whether to train on input tokens. Default is True.
    data_path (str, optional):
        The path to the dataset file. Default is "medical_meadow_small.json".
    train_in_8bit (bool, optional):
        Whether to use 8-bit training. Default is True.
    use_lora (bool, optional):
        Whether to use the Lora method. Default is True.
    lora_r (int, optional):
        The Lora method's reduction factor. Default is 8.
    lora_alpha (int, optional):
        The Lora method's alpha parameter. Default is 16.
    lora_dropout (float, optional):
        The dropout rate for Lora. Default is 0.1.
    lora_target_modules (List[str], optional):
        The target modules for Lora. Default is ["q_proj","v_proj"].
    per_device_batch_size (int, optional):
        The batch size per device. Default is 2.
    num_epochs (int, optional):
        The number of epochs for training. Default is 3.
    learning_rate (float, optional):
        The learning rate for the optimizer. Default is 2e-5.
    global_batch_size (int, optional):
        The number of samples the model needs to see until the weights get updated.
        Default is 128.
    output_dir (str, optional):
        The directory to save the model and outputs. Default is "./output".
    save_total_limit (int, optional):
        The maximum number of saved checkpoints. Default is 3.
    eval_steps (int, optional):
        The number of steps between evaluations. Default is 200.
    device_map (str, optional):
        The device placement strategy. Default is "auto".
    group_by_length (bool, optional):
        Whether to group samples by length for batch construction. Default is False.
    wandb_run_name (str, optional):
        The run name for Weights & Biases logging. Default is "test".
    use_wandb (bool, optional):
        Whether to use Weights & Biases for logging. Default is False.
    wandb_project (str, optional):
        The Weights & Biases project name. Default is "medalpaca".
    optim (str, optional):
        The optimizer to use. Default is "adamw_torch".
    lr_scheduler_type (str, optional):
        The learning rate scheduler type. Default is "cosine".
    fp16 (bool, optional):
        Whether to use mixed precision training (FP16). Default is True.
    bf16 (bool, optional):
        Whether to use mixed precision training (BF16). Default is False.
    gradient_checkpointing (bool, optional):
        Whether to use gradient checkpointing during training to reduce memory footprint
    warmup_steps (int, optional):
        The number of steps for warmup. Default is 200.
    fsdp (str, optional):
        Fully Sharded Data Parallel strategy. Only active with distributed training.
        Default is "full_shard auto_wrap"
    fsdp_transformer_layer_cls_to_wrap (optiona, str):
        The model layer to wrap for fsdp. Default is "LlamaDecoderLayer".
    **kwargs:
        additional arguments passed to the transformers.TrainingArguments"""
    # adapt arguments
    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = global_batch_size // per_device_batch_size
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        if use_lora:
            # integer and mixed dtypes are not supported with fsdp
            fsdp, fsdp_transformer_layer_cls_to_wrap = "", None
    else:
        fsdp, fsdp_transformer_layer_cls_to_wrap = "", None

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    # perform some checks, to raise errors early
    if fp16 and bf16:
        raise ValueError("At most one of fp16 and bf16 can be True, but not both.")

    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")

    # init model
    if "llama" in model_name:
        # The LLaMA config on HF is not up to date with the library,
        # leading to errors when using AutoModelForCausalLM
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    # loading the model with torch_dtype=torch.float16 with only fp16 and no LoRA leads
    # to `ValueError: Attempting to unscale FP16 gradients.`

    model = load_model.from_pretrained(
        model_name,
        load_in_8bit=train_in_8bit,
        torch_dtype=torch.float16 if any([use_lora, bf16]) else torch.float32,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    if train_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # init tokenizer and tokenize function
    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # load and tokenize data
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )
    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # init trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=eval_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        fsdp=fsdp,
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # for whatever reason, it is important that this is executed after trainer
    # is initialized. Otherwise you run into data indexing error, as the
    # trainer drops all columns in the dataset

    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # finally, train
    trainer.train()
    # Save the model
    model.save_pretrained(output_dir)

    # Evaluate and save predictions
    if val_set_size > 0:

        inferer = Inferer(
            model=model,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            base_model=None,
            model_max_length=model_max_length,
            load_in_8bit=False,
            torch_dtype=torch.float16
        )

        # 设置采样策略
        sampling_config = {
            "max_new_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.1,
            "top_k": 1,
            "num_beams": 1,
            "do_sample": False,
            "repetition_penalty": 1.0
        }
        
        # 评估模型
        predictions = evaluate_model(
            inferer=inferer,
            eval_data=data["test"],
            ntries=3,
            sampling=sampling_config
        )
        
        # 计算评估指标
        metrics = calculate_metrics(predictions)
        
        # 打印评估结果
        print("\nEvaluation Results:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Valid Predictions: {metrics['valid_ratio']*100:.1f}%")
        
        # 保存预测结果和指标
        results = {
            "predictions": predictions,
            "metrics": metrics
        }
        
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    fire.Fire(main)
