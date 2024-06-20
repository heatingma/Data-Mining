from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import numpy as np

model_name = 'bert-base-chinese'  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) 

def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

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

labels = np.where(data[:, 4].astype(int) > 3, 1, 0)

texts = data[:, 2]
labels = labels.astype(int)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


train_encodings = encode_texts(X_train.tolist())
test_encodings = encode_texts(X_test.tolist())
print('encode over')

train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(3):  
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


    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # 使用softmax计算概率
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # 选择正类（label=1）的概率
            predictions = probabilities[:, 1]
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算AUC值
    auc = roc_auc_score(all_labels, all_predictions)
    print(f'Test AUC: {auc:.2f}')