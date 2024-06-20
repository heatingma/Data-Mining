# RNN.py
from data_preprocess import get_data_loaders
from RNN_model import RNNClassifier
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

train_dataloader, test_dataloader, vocab = get_data_loaders(batch_size=8)

num_classes = 1 
vocab_size = len(vocab)  
embed_dim = 64
hidden_dim = 128

model = RNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def train(dataloader, model, criterion, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (text, label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label.squeeze(1), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += ((predicted_label.squeeze(1) > 0.5) == label).sum().item()
        total_count += label.size(0)
    return total_acc/total_count

def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label.squeeze(1), label)
            probabilities = torch.sigmoid(predicted_label).squeeze(1)
            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            total_acc += ((probabilities > 0.5) == label).sum().item()
            total_count += label.size(0)
    auc = roc_auc_score(all_labels, all_predictions)
    return total_acc/total_count, auc

if __name__ == "__main__":
    print('Train Start')
    num_epochs = 10  
    for epoch in range(num_epochs):
        accu_train = train(train_dataloader, model, criterion, optimizer)
        print(f'Epoch: {epoch+1}, Train accuracy: {accu_train:.2f}')
        
        accu_test, auc_test = evaluate(test_dataloader, model, criterion)
        print(f'Test accuracy: {accu_test:.2f}, Test AUC: {auc_test:.2f}')