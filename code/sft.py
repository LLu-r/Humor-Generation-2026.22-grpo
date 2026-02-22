"""
步骤2: 用Unsloth + LoRA + SFT训练Actor模型 (中文幽默生成)

任务对齐:
- 输入: headline 或 two words
- 输出: 幽默文本

数据格式: headline, word1, word2, joke
"""
import torch
import os
from unsloth import FastLanguageModel
from datasets import Dataset
from tqdm import tqdm
import pandas as pd

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============ 配置 ============
MODEL_NAME = 'unsloth/Qwen3-1.7B'  # 使用Qwen3-1.7B
OUTPUT_DIR = 'model/zh_actor_sft'
DATA_FILE = '../pre_data/zh_humor_with_prompts.csv'

MAX_LENGTH = 256  # 增加长度以容纳prompt+生成
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ============ 加载数据 ============
print("=" * 80)
print("步骤1: 加载数据")
print("=" * 80)

df = pd.read_csv(DATA_FILE, encoding='utf-8')
print(f"数据文件: {DATA_FILE}")
print(f"总样本数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 显示示例
print("\n数据示例:")
for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"\n样本 {i+1}:")
    print(f"  Headline: {row['headline']}")
    print(f"  Words: {row['word1']}, {row['word2']}")
    print(f"  Joke: {row['joke']}")

# ============ 构建训练数据 ============
print("\n" + "=" * 80)
print("步骤2: 构建训练数据")
print("=" * 80)

# Prompt模板
PROMPT_TEMPLATE_HEADLINE = """请根据以下新闻标题，创作一段幽默的中文文本：

标题：{headline}

幽默文本："""

PROMPT_TEMPLATE_WORDS = """请使用以下两个词语，创作一段幽默的中文文本：

词语：{word1}、{word2}

幽默文本："""

train_data = []

for idx, row in df.iterrows():
    headline = row['headline']
    word1 = row['word1']
    word2 = row['word2']
    joke = row['joke']
    
    # 方式1: 使用headline作为输入
    prompt_headline = PROMPT_TEMPLATE_HEADLINE.format(headline=headline)
    train_data.append({
        'prompt': prompt_headline,
        'completion': joke,
        'type': 'headline'
    })
    
    # 方式2: 使用two words作为输入
    prompt_words = PROMPT_TEMPLATE_WORDS.format(word1=word1, word2=word2)
    train_data.append({
        'prompt': prompt_words,
        'completion': joke,
        'type': 'words'
    })

print(f"训练样本数: {len(train_data)}")
print(f"  - 基于headline: {len([d for d in train_data if d['type'] == 'headline'])}")
print(f"  - 基于words: {len([d for d in train_data if d['type'] == 'words'])}")

# 转换为Dataset
dataset = Dataset.from_list(train_data)

# ============ 加载模型 ============
print("\n" + "=" * 80)
print("步骤3: 加载模型")
print("=" * 80)

print(f"加载模型: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# 添加LoRA
print("添加LoRA适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("模型加载完成！")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# ============ 数据处理函数 ============
def collator(batch):
    """将prompt和completion拼接,构建训练数据"""
    prompts = [item['prompt'] for item in batch]
    completions = [item['completion'] for item in batch]
    
    # 拼接prompt和completion，并在completion后添加EOS token
    # 这样模型会学习到在生成完成后输出EOS，从而自动停止生成
    texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    ).to(device)
    
    # 构建labels (只计算completion部分的loss，包括EOS token)
    labels = encodings['input_ids'].clone()
    
    # 对每个样本,mask掉prompt部分
    for i, (prompt, text) in enumerate(zip(prompts, texts)):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)
        labels[i, :prompt_len] = -100  # mask prompt部分
    
    # Mask padding tokens (但不mask EOS token)
    labels[labels == tokenizer.pad_token_id] = -100
    
    encodings['labels'] = labels
    return encodings


# ============ 创建DataLoader ============
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=collator
)

print(f"DataLoader批次数: {len(loader)}")

# ============ 优化器 ============
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ============ 训练 ============
print("\n" + "=" * 80)
print("步骤4: 开始训练")
print("=" * 80)

model.train()
global_step = 0

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*80}")
    
    pbar = tqdm(loader, desc=f"训练中")
    total_loss = 0
    valid_batches = 0
    
    for batch_idx, batch in enumerate(pbar):
        try:
            outputs = model(**batch)
            loss = outputs.loss
            
            # 检查NaN
            if torch.isnan(loss):
                print(f"\n[警告] 批次 {batch_idx} 出现NaN loss，跳过")
                optimizer.zero_grad()
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item()
            valid_batches += 1
            global_step += 1
            
            # 更新进度条
            avg_loss = total_loss / valid_batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            
        except Exception as e:
            print(f"\n[错误] 批次 {batch_idx} 训练失败: {e}")
            optimizer.zero_grad()
            continue
    
    # Epoch统计
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    print(f"\nEpoch {epoch+1} 完成:")
    print(f"  平均Loss: {avg_loss:.4f}")
    print(f"  有效批次: {valid_batches}/{len(loader)}")

# ============ 保存模型 ============
print("\n" + "=" * 80)
print("步骤5: 保存模型")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存LoRA权重
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ 模型已保存到: {OUTPUT_DIR}")
print(f"  - LoRA权重")
print(f"  - Tokenizer")

# ============ 测试生成 ============
print("\n" + "=" * 80)
print("步骤6: 测试生成")
print("=" * 80)

model.eval()

test_prompts = [
    PROMPT_TEMPLATE_HEADLINE.format(headline="学生上课迟到被老师批评"),
    PROMPT_TEMPLATE_WORDS.format(word1="吃", word2="蔬菜"),
]

print("测试模型是否学会自动停止生成（遇到EOS token）\n")

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"测试 {i+1}:")
    print(f"{'='*60}")
    print(f"输入Prompt:")
    print(f"  {prompt.strip()}")
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    print(f"\nPrompt token数: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_joke = generated_text[len(prompt):].strip()
    
    # 检查是否包含EOS token（在skip_special_tokens=False时）
    generated_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
    has_eos = tokenizer.eos_token in generated_with_special
    
    print(f"\n生成的幽默文本:")
    print(f"  {generated_joke}")
    print(f"\n生成token数: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
    print(f"是否生成EOS token: {'✅ 是' if has_eos else '❌ 否（可能达到max_new_tokens限制）'}")
    print("-" * 60)

print("\n" + "=" * 80)
print("SFT训练完成！")
print("=" * 80)
