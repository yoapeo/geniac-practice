import time
import wandb
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


model_id = "Qwen/Qwen3.5-0.8B"
question = "ノート1冊は120円です。3冊買うと何円ですか？短く答えてください。"

# MacのGPUが使えればMPSを選ぶ
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print("使用デバイス:", device)


# W&Bで、この実行を1つの実験（run）として記録する
run = wandb.init(
    entity="yoapeo-test",
    project="geniac-practice",
    job_type="inference",
    config={
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
    },
)
# トークナイザーとモデルを読み込む
print("トークナイザーを読み込みます...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("モデルを読み込みます。初回はダウンロードされます...")
model = Qwen3_5ForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation="sdpa",
)
model = model.to(device)
model.eval()

# 質問を会話形式に整え、トークンIDに変換する
messages = [{"role": "user", "content": question}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

# 回答を生成する
print("\n質問:", question)
print("回答を生成しています...")
torch.manual_seed(42)
start = time.perf_counter()

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

if device == "mps":
    torch.mps.synchronize()

elapsed = time.perf_counter() - start

# 入力した質問を除き、新しく生成された部分を取り出す
input_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[0, input_length:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n回答:", answer.strip())
print(f"生成時間: {elapsed:.2f}秒")

# 質問と回答を、表として保存する
examples = wandb.Table(
    columns=["question", "answer"],
    data=[[question, answer.strip()]],
)

# 生成時間・トークン数・回答の表を記録する
run.log({
    "generation_time_seconds": elapsed,
    "input_tokens": input_length,
    "output_tokens": generated_ids.numel(),
    "examples": examples,
})

result_url = run.url

# 記録を終了し、送信を完了する
run.finish()

print("W&Bの実行結果:", result_url)