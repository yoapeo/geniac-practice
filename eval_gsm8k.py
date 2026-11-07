import argparse
import random
import re
import time
from decimal import Decimal

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM


MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
VALIDATION_SIZE = 100
SPLIT_SEED = 42
MAX_NEW_TOKENS = 512
ENTITY = "yoapeo-test"
INSTRUCTION = (
    "Solve the problem briefly. On the last line, write #### followed by "
    "the final answer as a number only."
)


def final_number(text):
    """最後の行が #### 数値 なら、その数値を返す。"""
    lines = text.strip().splitlines()
    if not lines:
        return None

    number = r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?"
    match = re.fullmatch(rf"####[ \t]*({number})", lines[-1].strip())
    if match is None:
        return None
    return Decimal(match.group(1).replace(",", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, choices=(10, VALIDATION_SIZE), default=10)
    args = parser.parse_args()

    train = load_dataset(DATASET_ID, "main", split="train", revision=DATASET_REVISION)
    # trainの行番号を100個選んで検証用に確保する。
    validation_rows = random.Random(SPLIT_SEED).sample(range(len(train)), VALIDATION_SIZE)
    eval_rows = validation_rows[:args.limit]
    examples = train.select(eval_rows)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"{device}でモデルを読み込みます", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": "main/train",
        "split_seed": SPLIT_SEED,
        "validation_rows": validation_rows,
        "num_examples": len(examples),
        "prompt_instruction": INSTRUCTION,
        "enable_thinking": False,
        "do_sample": generation.do_sample,
        "max_new_tokens": generation.max_new_tokens,
        "device": device,
        "dtype": str(dtype),
    }

    with wandb.init(
        entity=ENTITY,
        project="geniac-practice",
        job_type="baseline-eval",
        name=f"gsm8k-baseline-{args.limit}",
        config=config,
    ) as run:
        predictions = wandb.Table(columns=[
            "train_row", "question", "gold", "response",
            "prediction", "correct", "output_tokens",
        ])
        correct_count = 0
        format_errors = 0
        start = time.perf_counter()

        with torch.inference_mode():
            for step, (row_index, example) in enumerate(zip(eval_rows, examples, strict=True), start=1):
                question = example["question"]
                gold = final_number(example["answer"])
                if gold is None:
                    raise ValueError(f"trainの行 {row_index} の正解を読み取れません")

                messages = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                print(f"{step}/{args.limit} 問目を生成中", flush=True)
                output = model.generate(**inputs, generation_config=generation)
                new_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
                response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                prediction = final_number(response)
                correct = prediction == gold
                correct_count += int(correct)
                format_errors += int(prediction is None)
                predictions.add_data(
                    row_index, question, str(gold), response,
                    "" if prediction is None else str(prediction), correct, len(new_ids),
                )

        accuracy = correct_count / len(examples)
        run.log({
            "accuracy": accuracy,
            "num_correct": correct_count,
            "num_examples": len(examples),
            "format_errors": format_errors,
            "eval_seconds": time.perf_counter() - start,
            "predictions": predictions,
        })
        run_url = run.url

    print(f"正答率 {correct_count}/{len(examples)} = {accuracy:.1%}")
    print(f"形式エラー {format_errors}問")
    print("W&Bの実行結果", run_url)


if __name__ == "__main__":
    main()