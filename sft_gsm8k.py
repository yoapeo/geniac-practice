import argparse
import math
import random
import re
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from eval_gsm8k import (
    DATASET_ID,
    DATASET_REVISION,
    ENTITY,
    INSTRUCTION,
    MODEL_ID,
    MODEL_REVISION,
    SPLIT_SEED,
    VALIDATION_SIZE,
    final_number,
)


TRAIN_SEED = 43
MAX_SEQUENCE_LENGTH = 256
LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def prepare_examples(train, tokenizer, steps):
    # 評価と同じ規則で100問を確保し、学習候補から除く。
    validation_rows = set(
        random.Random(SPLIT_SEED).sample(range(len(train)), VALIDATION_SIZE)
    )
    candidate_rows = [i for i in range(len(train)) if i not in validation_rows]
    random.Random(TRAIN_SEED).shuffle(candidate_rows)

    examples = []
    for row in candidate_rows:
        example = train[row]
        messages = [
            {"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 計算用注記を外し、解説と最終行の #### 数値を残す。
        answer = re.sub(r"<<[^<>]*>>", "", example["answer"]).strip()
        if final_number(answer) is None:
            raise ValueError(f"trainの行 {row} の正解を読み取れません")

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > MAX_SEQUENCE_LENGTH:
            continue

        labels = [-100] * len(prompt_ids) + answer_ids
        examples.append((row, input_ids, labels))
        if len(examples) == steps:
            return examples

    raise ValueError(f"条件に合う学習例が {len(examples)} 問しかありません")


def add_lora(model):
    last_layer = model.config.num_hidden_layers - 1
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=[last_layer],
        layers_pattern="layers",
        bias="none",
    )
    return get_peft_model(model, config, revision=MODEL_REVISION)


def train_step(model, optimizer, input_ids, labels, device):
    batch = {
        "input_ids": torch.tensor([input_ids], device=device),
        "labels": torch.tensor([labels], device=device),
    }
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    value = loss.item()
    if not math.isfinite(value):
        raise RuntimeError("lossが有限値になりませんでした")
    loss.backward()
    optimizer.step()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps には1以上を指定してください")

    output = args.output or Path("checkpoints") / f"sft-{args.steps}"
    if output.exists():
        parser.error(f"{output} は既にあります。--output で別の保存先を指定してください")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MacのGPU (MPS) を利用できません。check_env.py を確認してください")

    device = "mps"
    torch.manual_seed(TRAIN_SEED)
    train = load_dataset(DATASET_ID, "main", split="train", revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = prepare_examples(train, tokenizer, args.steps)

    print("MPSでモデルを読み込みます", flush=True)
    base = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model = add_lora(base)
    model.train()
    model.print_trainable_parameters()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, foreach=False)

    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": "main/train",
        "split_seed": SPLIT_SEED,
        "validation_size": VALIDATION_SIZE,
        "train_seed": TRAIN_SEED,
        "train_rows": [row for row, _, _ in examples],
        "steps": args.steps,
        "batch_size": 1,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_layer": model.config.num_hidden_layers - 1,
        "lora_target_modules": LORA_TARGET_MODULES,
        "device": device,
        "base_dtype": "float16",
        "adapter_path": str(output),
    }
    with wandb.init(
        entity=ENTITY,
        project="geniac-practice",
        job_type="sft",
        name=output.name,
        config=config,
    ) as run:
        for step, (row, input_ids, labels) in enumerate(examples, start=1):
            loss = train_step(model, optimizer, input_ids, labels, device)
            print(f"{step}/{args.steps}  loss={loss:.4f}  train_row={row}", flush=True)
            run.log({"train/loss": loss, "train/row": row}, step=step)

        model.save_pretrained(output, save_embedding_layers=False)
        run_url = run.url

    print(f"LoRAを保存しました {output}")
    print("W&Bの実行結果", run_url)


if __name__ == "__main__":
    main()