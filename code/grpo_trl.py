"""
步骤3: GRPO训练 (中文幽默生成)

GRPO设计:
- Policy: SFT后的Qwen3-4B (可训练, 4-bit + LoRA)
- RM: 训练好的DeBERTa打分模型 (frozen, 用于计算reward)

Reward流程:
1. Qwen3生成 → token ids
2. Actor tokenizer解码 → 中文文本字符串
3. DeBERTa tokenizer编码 → DeBERTa的token ids
4. DeBERTa模型打分 → reward分数
"""
import torch
import os
import shutil
import pandas as pd
from datasets import Dataset
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig
)
from trl import ModelConfig, get_peft_config, GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training  # 添加PEFT导入

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============ 配置 ============
# 模型路径
BASE_MODEL_PATH = 'unsloth/Qwen3-1.7B'  # 基础模型 (与SFT训练时相同)
SFT_LORA_PATH = 'model/zh_actor_sft'  # SFT训练的LoRA权重
RM_MODEL_PATH = '../../humor_ppo_qwen3-4b-true/code/new_code2.12/code_zh/RM/model/zh_reward_model_deberta'
RM_TOKENIZER_PATH = 'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese'

# 定义合并后模型的保存路径
MERGED_MODEL_DIR = "model/qwen3_1.7b_sft_merged"

# 数据路径
TRAIN_DATA_FILE = '../pre_data/zh_humor_with_prompts.csv'
TEST_DATA_FILE = '../../humor_ppo_qwen3-4b-true/pre_data/humor_only_with_prompts/zh_humor_with_prompts_test.csv'

# 输出路径
OUTPUT_DIR = 'model/zh_grpo_trl'

# GRPO训练参数
LEARNING_RATE = 1e-6
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 100

# GRPO特定参数
NUM_SAMPLE_GENERATIONS = 4  # 每个prompt生成的样本数（组大小）
generation_batch_size = 64  #生成批次大小 等于NUM_SAMPLE_GENERATIONS*batch_size
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# Prompt模板
PROMPT_TEMPLATE_HEADLINE = """请根据以下新闻标题，创作一段幽默的中文文本：

标题：{headline}

幽默文本："""

PROMPT_TEMPLATE_WORDS = """请使用以下两个词语，创作一段幽默的中文文本：

词语：{word1}、{word2}

幽默文本："""

# 清理输出目录
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

print("=" * 80)
print("GRPO训练 (中文幽默生成)")
print("=" * 80)
print(f"基础模型: {BASE_MODEL_PATH}")
print(f"SFT LoRA权重: {SFT_LORA_PATH}")
print(f"RM模型: {RM_MODEL_PATH}")
print(f"训练数据: {TRAIN_DATA_FILE}")
print(f"测试数据: {TEST_DATA_FILE}")
print(f"输出目录: {OUTPUT_DIR}")


# ============ 加载数据 ============
print("\n" + "=" * 80)
print("步骤1: 加载训练和测试数据")
print("=" * 80)

# 加载训练数据
train_df = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8')
print(f"训练数据文件: {TRAIN_DATA_FILE}")
print(f"训练样本数: {len(train_df)}")

#############################只使用前1000条数据
train_df = train_df.head(1000)
print(f" 截断完成：只使用前 {len(train_df)} 条原始数据进行训练")

# 加载测试数据
test_df = pd.read_csv(TEST_DATA_FILE, encoding='utf-8')
print(f"\n测试数据文件: {TEST_DATA_FILE}")
print(f"测试样本数: {len(test_df)}")

#测试集也建议截断一下，比如只用前 100 条，防止评估阶段等太久
test_df = test_df.head(100)
print(f"✅ 截断完成：只使用前 {len(test_df)} 条数据进行测试")

# 显示训练数据示例
print("\n训练数据示例:")
for i in range(min(3, len(train_df))):
    row = train_df.iloc[i]
    print(f"\n样本 {i+1}:")
    print(f"  Headline: {row['headline']}")
    print(f"  Words: {row['word1']}, {row['word2']}")
    print(f"  Joke: {row['joke'][:50]}...")

# ============ 构建训练和测试数据 ============
print("\n" + "=" * 80)
print("步骤2: 构建训练和测试数据")
print("=" * 80)

# ⚠️ 关键修改: GRPO需要列名为'prompt'而不是'query'
# 并且不需要预先分词,直接使用文本
train_data = []
for idx, row in train_df.iterrows():
    headline = row['headline']
    word1 = row['word1']
    word2 = row['word2']
    
    # 方式1: 使用headline
    prompt_headline = PROMPT_TEMPLATE_HEADLINE.format(headline=headline)
    train_data.append({"prompt": prompt_headline})  # 列名改为'prompt'
    
    # 方式2: 使用words
    prompt_words = PROMPT_TEMPLATE_WORDS.format(word1=word1, word2=word2)
    train_data.append({"prompt": prompt_words})  # 列名改为'prompt'

print(f"训练Prompt总数: {len(train_data)}")
print(f"  - 基于headline: {len([q for i, q in enumerate(train_data) if i % 2 == 0])}")
print(f"  - 基于words: {len([q for i, q in enumerate(train_data) if i % 2 == 1])}")

# 构建测试数据
test_data = []
for idx, row in test_df.iterrows():
    headline = row['headline']
    word1 = row['word1']
    word2 = row['word2']
    
    # 方式1: 使用headline
    prompt_headline = PROMPT_TEMPLATE_HEADLINE.format(headline=headline)
    test_data.append({"prompt": prompt_headline})  # 列名改为'prompt'
    
    # 方式2: 使用words
    prompt_words = PROMPT_TEMPLATE_WORDS.format(word1=word1, word2=word2)
    test_data.append({"prompt": prompt_words})  # 列名改为'prompt'

print(f"\n测试Prompt总数: {len(test_data)}")
print(f"  - 基于headline: {len([q for i, q in enumerate(test_data) if i % 2 == 0])}")
print(f"  - 基于words: {len([q for i, q in enumerate(test_data) if i % 2 == 1])}")

# 显示训练prompt示例
print("\n训练Prompt示例:")
for i in range(min(3, len(train_data))):
    print(f"\n示例 {i+1}:")
    print(train_data[i]['prompt'].strip())

# 转换为Dataset - 直接使用文本,不进行分词
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(test_data)

print(f"\nDataset创建完成:")
print(f"  - 训练集: {len(train_dataset)} 条 (来自独立训练数据)")
print(f"  - 评估集: {len(eval_dataset)} 条 (来自独立测试数据)")
print(f"  ⚠️ 注意: 数据集保持文本格式,不进行预分词")


# ============ 加载Tokenizer ============
print("\n" + "=" * 80)
print("步骤3: 加载Tokenizer")
print("=" * 80)

# 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Policy tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,  # 使用基础模型的tokenizer
    padding_side="left",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Policy分词器加载完成")
print(f"  - Vocab size: {len(tokenizer)}")
print(f"  - Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# RM tokenizer
rm_tokenizer = AutoTokenizer.from_pretrained(RM_TOKENIZER_PATH)
print("\nRM分词器加载完成")
print(f"  - Vocab size: {len(rm_tokenizer)}")


# ============ 加载Policy模型 (基础模型 + SFT LoRA) ============
print("\n" + "=" * 80)
print("步骤4: 准备Policy模型 (离线合并 SFT LoRA + 4bit加载)")
print("=" * 80)

# 4.1 检查是否已经合并过。如果没有，则以 bfloat16 (不使用4bit) 加载并合并
if not os.path.exists(MERGED_MODEL_DIR):
    print(f"4.1 首次运行：以 bfloat16 加载基础模型进行合并 (不量化)")
    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 使用 CPU 合并以节省显存，如果显卡显存充裕也可以用 "auto"
        trust_remote_code=True,
    )
    
    print(f"4.2 加载SFT LoRA并合并: {SFT_LORA_PATH}")
    sft_model = PeftModel.from_pretrained(base_model_for_merge, SFT_LORA_PATH)
    merged_model = sft_model.merge_and_unload()
    
    print(f"4.3 保存合并后的模型到: {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    
    # 彻底清理内存和显存，为后续的 4bit GRPO 训练腾出空间
    del base_model_for_merge, sft_model, merged_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("  ✓ 合并完成并已释放内存！")
else:
    print(f"4.1 发现已合并的模型: {MERGED_MODEL_DIR}，跳过合并步骤。")

# 4.4 将合并好的模型当作全新的"基础模型"，以 4bit 方式加载
print(f"\n4.4 以 4bit 量化方式加载合并后的新模型，准备 GRPO 训练")
policy = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_DIR,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 4.5 开启梯度并手动挂载 GRPO 专属 LoRA
print(f"\n4.5 开启梯度并手动挂载 GRPO 专属 LoRA")

if hasattr(policy, "enable_input_require_grads"):
    policy.enable_input_require_grads()

policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

grpo_lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type="CAUSAL_LM"
)
policy = get_peft_model(policy, grpo_lora_config)

# 🚀 终极修复 1：模块级别的类型强转 (比修改 param.data 更彻底，只转化非量化层)
for name, module in policy.named_modules():
    # 匹配标准的线性层 (如 lm_head)。注意：这不会影响 4bit 量化层 (bnb.nn.Linear4bit)
    if isinstance(module, torch.nn.Linear):
        module.to(torch.bfloat16)
    # 将 LayerNorm 和 Embedding 也统一转化
    if "norm" in name.lower() or "embed" in name.lower():
        module.to(torch.bfloat16)

# 🚀 终极修复 2：猴子补丁 (Monkey Patch)，给 generate 方法套上硬件级的 BFloat16 强制上下文
original_generate = policy.generate
def autocast_generate(*args, **kwargs):
    # 强制让生成过程在 BFloat16 的 Autocast 环境下运行，杜绝一切类型不匹配！
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return original_generate(*args, **kwargs)
policy.generate = autocast_generate

print("  ✓ Policy模型加载完成 (基础模型 + SFT + 4bit量化 + BFloat16模块级对齐 + Autocast保护 + GRPO LoRA)")
policy.print_trainable_parameters()


# ============ 加载Reward Model ============
print("\n" + "=" * 80)
print("步骤5: 加载Reward Model")
print("=" * 80)

# 加载DeBERTa奖励模型
deberta_rm = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
deberta_rm.eval()
for param in deberta_rm.parameters():
    param.requires_grad = False

print("DeBERTa RM模型加载完成")

# ⚠️ 关键修改: GRPO不需要预先分词,直接使用文本数据集
# 删除了原来的数据预处理步骤

# ============ 配置GRPO训练参数 ============
print("\n" + "=" * 80)
print("步骤6: 配置GRPO训练参数")
print("=" * 80)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # GRPO核心参数
    num_generations=NUM_SAMPLE_GENERATIONS,  # 组大小G
    generation_batch_size=generation_batch_size,  # 必须能被num_generations整除
    max_prompt_length=256,  # 必须设置: Prompt最大长度
    max_completion_length=MAX_NEW_TOKENS,  # 必须设置: 生成内容最大长度
    
    # 算法参数
    beta=0.05,  # KL散度惩罚系数
    temperature=TEMPERATURE,
    
    # 运行优化
    bf16=True,  # 如果硬件支持,建议开启
    remove_unused_columns=False,  # 重要: 防止删掉prompt列
    
    # 优化参数
    max_grad_norm=1.0,
    warmup_steps=50,
)

print("GRPO训练参数:")
print(f"  - 学习率: {LEARNING_RATE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - 训练轮数: {EPOCHS}")
print(f"  - 组大小 (num_generations): {NUM_SAMPLE_GENERATIONS}")
print(f"  - 生成批次大小 (generation_batch_size): {NUM_SAMPLE_GENERATIONS}")
print(f"  - Max prompt length: 256")
print(f"  - Max completion length: {MAX_NEW_TOKENS}")
print(f"  - Beta (KL penalty): 0.04")
print(f"  - 温度: {TEMPERATURE}")
print(f"  - BF16: True")
print(f"  - Remove unused columns: False")
print(f"  - TensorBoard日志: {OUTPUT_DIR}/logs")

# ============ 配置LoRA ============
print("\n" + "=" * 80)
print("步骤8: 配置LoRA")
print("=" * 80)

model_args = ModelConfig(
    model_name_or_path=BASE_MODEL_PATH,  # 使用基础模型路径
    load_in_4bit=True,
    trust_remote_code=True,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "down_proj", "up_proj"
    ]
)

peft_config = get_peft_config(model_args)

print("LoRA配置:")
print(f"  - LoRA rank (r): {LORA_R}")
print(f"  - LoRA alpha: {LORA_ALPHA}")
print(f"  - LoRA dropout: {LORA_DROPOUT}")
print(f"  - 目标模块: {model_args.lora_target_modules}")

# ============ 定义Reward Function ============
def reward_function(prompts, completions, **kwargs):
    """
    GRPO要求的reward function格式
    
    Args:
        prompts: 输入的prompt列表
        completions: 生成的completion列表
        **kwargs: 其他参数
    
    Returns:
        rewards: 每个completion的reward分数列表
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # 组合prompt + completion
        full_text = prompt + completion
        
        # 使用DeBERTa RM评分
        rm_inputs = rm_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(deberta_rm.device)
        
        with torch.no_grad():
            outputs = deberta_rm(**rm_inputs)
            # 取幽默logit作为reward
            reward = outputs.logits[0, 1].item()
        
        rewards.append(reward)
    
    return rewards

print("\nReward Function定义完成")

# ============ 创建GRPO训练器 ============
print("\n" + "=" * 80)
print("步骤8: 创建GRPO训练器")
print("=" * 80)

trainer = GRPOTrainer(
    model=policy,  # 传入上面已经手动挂载好 LoRA 的 PeftModel
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_funcs=reward_function,
    # ⚠️ 关键：这里去掉了 peft_config 参数，防止 TRL 重复包装导致报错
)

print("GRPO训练器创建完成")
print("  - 模型: 基础模型 + SFT (已合并) + 针对 GRPO 的新 LoRA")

# ============ 开始训练 ============
print("\n" + "=" * 80)
print("步骤9: 开始GRPO训练")
print("=" * 80)
print("=" * 50)

trainer.train()

print("=" * 50)
print("GRPO训练完成!")

# ============ 保存模型 ============
print("\n" + "=" * 80)
print("步骤10: 保存模型")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)

print(f"✅ 模型已保存到: {OUTPUT_DIR}")


# ============ 测试生成 ============
print("\n" + "=" * 80)
print("步骤11: 测试生成")
print("=" * 80)

# 加载训练好的模型
trained_policy = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)
print("训练好的模型加载完成")

# 配置生成参数
generation_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# 测试prompt
test_prompts = [
    PROMPT_TEMPLATE_HEADLINE.format(headline="学生上课迟到被老师批评"),
    PROMPT_TEMPLATE_WORDS.format(word1="吃", word2="蔬菜"),
    PROMPT_TEMPLATE_HEADLINE.format(headline="程序员加班到深夜"),
]

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"测试 {i+1}:")
    print(f"{'='*60}")
    print(f"输入Prompt:")
    print(f"  {prompt.strip()}")
    
    inputs = tokenizer(prompt, return_tensors='pt').to(trained_policy.device)
    print(f"\nPrompt token数: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = trained_policy.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_joke = generated_text[len(prompt):].strip()
    
    print(f"\n生成的幽默文本:")
    print(f"  {generated_joke}")
    print(f"\n生成token数: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
    
    # 使用RM评分
    rm_inputs = rm_tokenizer(
        generated_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(deberta_rm.device)
    
    with torch.no_grad():
        rm_outputs = deberta_rm(**rm_inputs)
        logits = rm_outputs.logits
        probs = torch.softmax(logits, dim=-1)
        humor_score = probs[0, 1].item()
        humor_logit = logits[0, 1].item()
    
    print(f"\nRM评分:")
    print(f"  不幽默概率: {probs[0, 0].item():.4f}")
    print(f"  幽默概率: {humor_score:.4f}")
    print(f"  幽默logit: {humor_logit:.4f}")
    print("-" * 60)

# ============ 训练总结 ============
print("\n" + "=" * 80)
print("训练完成总结")
print("=" * 80)
print(f"训练数据量: {len(train_dataset)}")
print(f"每个prompt生成样本数: {NUM_SAMPLE_GENERATIONS}")
print(f"模型保存位置: {OUTPUT_DIR}")
print(f"TensorBoard日志: {OUTPUT_DIR}/logs")
print(f"\n查看训练曲线:")
print(f"  tensorboard --logdir={OUTPUT_DIR}/logs")

print("\n" + "=" * 80)
print("GRPO训练完成！")
print("=" * 80)

