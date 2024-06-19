from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
import torch
import numpy as np

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-chinese'  # 对于中文文本，使用中文预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类问题

# 将文本编码为模型输入格式
def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# 数据集类
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
    
data = np.load('../cache/movie_comments.npy')

# 将评分转换为二分类标签
labels = np.where(data[:, 4].astype(int) > 3, 1, 0)

# 文本内容和标签
texts = data[:, 2]
labels = labels.astype(int)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 编码训练和测试文本
train_encodings = encode_texts(X_train.tolist())
test_encodings = encode_texts(X_test.tolist())
print('encode over')
# 创建数据集和数据加载器
train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(10):  # 训练3个epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


    # 评估模型
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs.logits.argmax(dim=-1)
            total_acc += (predictions == labels).sum().item()
            print(predictions)
            print(labels)
            print(total_count)
            total_count += labels.size(0)
    print(f'Test accuracy: {total_acc / total_count:.2f}')