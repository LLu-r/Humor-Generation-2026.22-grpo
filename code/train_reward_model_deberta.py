"""
使用 Erlangshen-DeBERTa-v2-97M-Chinese + R-drop 训练 Reward Model

优势：
- 模型更小（97M vs 4B），不容易过拟合
- 专门为分类任务设计
- 使用 R-drop 提高泛化能力
- 训练更快，效果更好
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ======================
# 配置
# ======================
MODEL_NAME = 'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese'  # 二郎神中文DeBERTa
TRAIN_DATA_FILE = "../../../pre_data/train_RM_data/train_RM_data_zh.csv"
TEST_DATA_FILE = "../../../pre_data/test_RM_data/test_RM_data_zh.csv"
OUTPUT_DIR = "model/zh_reward_model_deberta"

MAX_LENGTH = 128
BATCH_SIZE = 16  # DeBERTa 更小，可以用更大的 batch size
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
RDROP_ALPHA = 4.0  # R-drop KL 散度权重

# Prompt 模板（与 SFT 一致）
PROMPT_TEMPLATE_HEADLINE = """请根据以下新闻标题，创作一段幽默的中文文本：

标题：{headline}

幽默文本："""

PROMPT_TEMPLATE_WORDS = """请使用以下两个词语，创作一段幽默的中文文本：

词语：{word1}、{word2}

幽默文本："""

print("\n" + "=" * 80)
print("Reward Model 训练 (DeBERTa + R-Drop)")
print("=" * 80)
print(f"模型: {MODEL_NAME}")
print(f"训练数据: {TRAIN_DATA_FILE}")
print(f"测试数据: {TEST_DATA_FILE}")
print(f"输出目录: {OUTPUT_DIR}")

# ======================
# 1. 加载数据
# ======================
print("\n" + "=" * 80)
print("步骤1: 加载数据")
print("=" * 80)

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

print(f"训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")

# 统计标签分布
train_label_counts = train_df['label'].value_counts()
test_label_counts = test_df['label'].value_counts()

print(f"\n训练集标签分布:")
print(f"  label=0 (不幽默): {train_label_counts.get(0, 0)} ({train_label_counts.get(0, 0)/len(train_df)*100:.1f}%)")
print(f"  label=1 (幽默): {train_label_counts.get(1, 0)} ({train_label_counts.get(1, 0)/len(train_df)*100:.1f}%)")

print(f"\n测试集标签分布:")
print(f"  label=0 (不幽默): {test_label_counts.get(0, 0)} ({test_label_counts.get(0, 0)/len(test_df)*100:.1f}%)")
print(f"  label=1 (幽默): {test_label_counts.get(1, 0)} ({test_label_counts.get(1, 0)/len(test_df)*100:.1f}%)")

# 显示数据示例
print("\n数据示例:")
for i in range(min(3, len(train_df))):
    row = train_df.iloc[i]
    print(f"\n样本 {i+1}:")
    print(f"  Headline: {row['headline']}")
    print(f"  Words: {row['word1']}, {row['word2']}")
    print(f"  Joke: {row['joke'][:50]}...")
    print(f"  Label: {row['label']} ({'幽默' if row['label'] == 1 else '不幽默'})")

# ======================
# 2. 构建输入文本
# ======================
print("\n" + "=" * 80)
print("步骤2: 构建输入文本")
print("=" * 80)

def build_input_text(row):
    """构建 prompt + joke 作为输入"""
    headline = str(row['headline']).strip()
    word1 = str(row['word1']).strip()
    word2 = str(row['word2']).strip()
    joke = str(row['joke']).strip()
    
    # 判断使用哪种 prompt 模板
    if headline and headline != '-' and headline.lower() != 'nan':
        prompt = PROMPT_TEMPLATE_HEADLINE.format(headline=headline)
    else:
        prompt = PROMPT_TEMPLATE_WORDS.format(word1=word1, word2=word2)
    
    # 拼接 prompt 和 joke
    full_text = prompt + joke
    return full_text

# 构建训练集文本
train_texts = []
train_labels = []
for idx, row in train_df.iterrows():
    try:
        text = build_input_text(row)
        train_texts.append(text)
        train_labels.append(int(row['label']))
    except Exception as e:
        print(f"跳过训练集第 {idx} 行: {e}")

# 构建测试集文本
test_texts = []
test_labels = []
for idx, row in test_df.iterrows():
    try:
        text = build_input_text(row)
        test_texts.append(text)
        test_labels.append(int(row['label']))
    except Exception as e:
        print(f"跳过测试集第 {idx} 行: {e}")

print(f"训练集处理完成: {len(train_texts)} 条")
print(f"测试集处理完成: {len(test_texts)} 条")

# ======================
# 3. Dataset 类
# ======================
class RewardModelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ======================
# 4. R-drop Loss
# ======================
def compute_kl_loss(p, q):
    """计算两个分布之间的对称 KL 散度"""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2


def rdrop_loss(logits1, logits2, labels, alpha=4.0):
    """R-drop loss = CE loss + alpha * KL loss"""
    ce_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2
    kl_loss = compute_kl_loss(logits1, logits2)
    return ce_loss + alpha * kl_loss

# ======================
# 5. 初始化模型
# ======================
print("\n" + "=" * 80)
print("步骤3: 初始化模型")
print("=" * 80)

print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 创建 Dataset 和 DataLoader
train_dataset = RewardModelDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
test_dataset = RewardModelDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"\n训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ======================
# 6. 优化器和调度器
# ======================
print("\n" + "=" * 80)
print("步骤4: 设置优化器")
print("=" * 80)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
num_warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=total_steps
)

print(f"总训练步数: {total_steps}")
print(f"Warmup步数: {num_warmup_steps}")
print(f"学习率: {LR}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"R-drop alpha: {RDROP_ALPHA}")

# ======================
# 7. 评估函数
# ======================
def evaluate(model, loader, dataset_name=""):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"评估{dataset_name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / len(loader)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    label0_acc = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
    label1_acc = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'label0_acc': label0_acc,
        'label1_acc': label1_acc,
        'confusion_matrix': cm
    }

# ======================
# 8. 训练循环
# ======================
print("\n" + "=" * 80)
print("步骤5: 开始训练")
print("=" * 80)

best_test_f1 = 0
best_test_acc = 0
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*80}")
    
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(train_loader, desc="训练中")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # R-drop: 前向传播两次（使用不同的 dropout mask）
        outputs1 = model(input_ids, attention_mask=attention_mask)
        outputs2 = model(input_ids, attention_mask=attention_mask)
        
        # 计算 R-drop loss
        ce_loss = (F.cross_entropy(outputs1.logits, labels) + F.cross_entropy(outputs2.logits, labels)) / 2
        kl_loss = compute_kl_loss(outputs1.logits, outputs2.logits)
        loss = ce_loss + RDROP_ALPHA * kl_loss
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kl_loss += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    # Epoch 统计
    avg_loss = total_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    
    print(f"\nEpoch {epoch+1} 训练完成:")
    print(f"  平均 Total Loss: {avg_loss:.4f}")
    print(f"  平均 CE Loss: {avg_ce_loss:.4f}")
    print(f"  平均 KL Loss: {avg_kl_loss:.4f}")
    
    # 在测试集上评估
    print(f"\n在测试集上评估...")
    test_metrics = evaluate(model, test_loader, "测试集")
    
    print(f"\n测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  准确率: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")
    print(f"  label=0 (不幽默) 准确率: {test_metrics['label0_acc']:.4f}")
    print(f"  label=1 (幽默) 准确率: {test_metrics['label1_acc']:.4f}")
    
    # 保存最佳模型（基于 F1 分数）
    if test_metrics['f1'] > best_test_f1:
        best_test_f1 = test_metrics['f1']
        best_test_acc = test_metrics['accuracy']
        
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # 保存评估指标
        metrics_file = os.path.join(OUTPUT_DIR, "best_metrics.txt")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
            f.write(f"Label=0 Accuracy: {test_metrics['label0_acc']:.4f}\n")
            f.write(f"Label=1 Accuracy: {test_metrics['label1_acc']:.4f}\n")
        
        print(f"  ✅ 新的最佳模型! F1: {best_test_f1:.4f}, Acc: {best_test_acc:.4f}")
        print(f"  已保存到: {OUTPUT_DIR}")

# ======================
# 9. 最终评估
# ======================
print("\n" + "=" * 80)
print("步骤6: 最终评估")
print("=" * 80)

# 加载最佳模型
print(f"加载最佳模型: {OUTPUT_DIR}")
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)

# 在测试集上最终评估
final_metrics = evaluate(model, test_loader, "最终测试集")

print(f"\n最终测试集结果:")
print(f"{'='*60}")
print(f"准确率 (Accuracy):  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
print(f"精确率 (Precision): {final_metrics['precision']:.4f}")
print(f"召回率 (Recall):    {final_metrics['recall']:.4f}")
print(f"F1分数 (F1-Score):  {final_metrics['f1']:.4f}")
print(f"{'='*60}")
print(f"label=0 (不幽默) 准确率: {final_metrics['label0_acc']:.4f}")
print(f"label=1 (幽默) 准确率: {final_metrics['label1_acc']:.4f}")
print(f"{'='*60}")

# 显示混淆矩阵
cm = final_metrics['confusion_matrix']
print(f"\n混淆矩阵:")
print(f"                预测: 不幽默    预测: 幽默")
print(f"实际: 不幽默        {cm[0][0]:6d}        {cm[0][1]:6d}")
print(f"实际: 幽默          {cm[1][0]:6d}        {cm[1][1]:6d}")

# 保存最终报告
report_file = os.path.join(OUTPUT_DIR, "training_report.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Reward Model 训练报告 (DeBERTa + R-Drop)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("## 模型配置\n")
    f.write(f"- 模型: {MODEL_NAME}\n")
    f.write(f"- 最大长度: {MAX_LENGTH}\n")
    f.write(f"- Batch size: {BATCH_SIZE}\n")
    f.write(f"- Epochs: {EPOCHS}\n")
    f.write(f"- 学习率: {LR}\n")
    f.write(f"- Weight decay: {WEIGHT_DECAY}\n")
    f.write(f"- R-drop alpha: {RDROP_ALPHA}\n\n")
    
    f.write("## 数据集\n")
    f.write(f"- 训练集: {len(train_texts)} 条\n")
    f.write(f"- 测试集: {len(test_texts)} 条\n\n")
    
    f.write("## 最终测试集结果\n")
    f.write(f"- 准确率: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)\n")
    f.write(f"- 精确率: {final_metrics['precision']:.4f}\n")
    f.write(f"- 召回率: {final_metrics['recall']:.4f}\n")
    f.write(f"- F1分数: {final_metrics['f1']:.4f}\n")
    f.write(f"- label=0 准确率: {final_metrics['label0_acc']:.4f}\n")
    f.write(f"- label=1 准确率: {final_metrics['label1_acc']:.4f}\n\n")
    
    f.write("## 混淆矩阵\n")
    f.write(f"                预测: 不幽默    预测: 幽默\n")
    f.write(f"实际: 不幽默        {cm[0][0]:6d}        {cm[0][1]:6d}\n")
    f.write(f"实际: 幽默          {cm[1][0]:6d}        {cm[1][1]:6d}\n")

print(f"\n✅ 训练报告已保存: {report_file}")

print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)
print(f"最佳模型已保存到: {OUTPUT_DIR}")
print(f"最佳 F1 分数: {best_test_f1:.4f}")
print(f"最佳准确率: {best_test_acc:.4f}")
print("\n下一步: 使用此模型作为 PPO 的 Reward Model")
