# necessary  libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from networks.simple_model import SimpleModel

# create a simple dataset

# Dummy dataset: 100 samples, 10 features each
X = torch.randn(100, 10)
y = (torch.randn(100) > 0).long()  # Binary targets
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# define a simple nerual network model



model = SimpleModel()

#set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#train the model

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#Evaluate the model and calculate the F1 score:
# Generate predictions
model.eval()  # Set the model to evaluation mode
all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_targets.extend(targets.numpy())
        all_predictions.extend(predicted.numpy())

# Calculate F1 score

f1 = f1_score(all_targets, all_predictions, average='binary')
print(f"F1 Score: {f1}")

micro_f1 = f1_score(all_targets, all_predictions, average='micro')
macro_f1 = f1_score(all_targets, all_predictions, average='macro')
weighted_f1 = f1_score(all_targets, all_predictions, average='weighted')

print(f"Micro F1 Score: {micro_f1}")
print(f"Macro F1 Score: {macro_f1}")
print(f"Weighted F1 Score: {weighted_f1}")

# calcualte the f1 score manually
# Calculate TP, FP, FN
TP = 0
FP = 0
FN = 0

for target, prediction in zip(all_targets, all_predictions):
    if target == 1 and prediction == 1:
        TP += 1
    elif target == 0 and prediction == 1:
        FP += 1
    elif target == 1 and prediction == 0:
        FN += 1

# Calculate precision and recall
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1 score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score (manual) : {f1}")


#Manually Calculating Different F1 Scores
# Initialize counters
num_classes = 2  # Binary classification

TP = np.zeros(num_classes)
FP = np.zeros(num_classes)
FN = np.zeros(num_classes)

# Calculate TP, FP, FN for each class
for target, prediction in zip(all_targets, all_predictions):
    if target == prediction:
        TP[target] += 1
    else:
        FP[prediction] += 1
        FN[target] += 1

# Calculate precision, recall, F1 for each class
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1_per_class = np.zeros(num_classes)

for i in range(num_classes):
    precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
    recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
    f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

# Calculate micro, macro, and weighted F1 scores
total_TP = np.sum(TP)
total_FP = np.sum(FP)
total_FN = np.sum(FN)

# micro_f1_manual = 2 * (np.sum(TP) * np.sum(TP) / (np.sum(TP) + np.sum(FP)) * np.sum(TP) / (np.sum(TP) + np.sum(FN))) / \
#     (np.sum(TP) / (np.sum(TP) + np.sum(FP)) + np.sum(TP) / (np.sum(TP) + np.sum(FN)))

micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0

micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

micro_f1_manual = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0


macro_f1_manual = np.mean(f1_per_class)

weighted_f1_manual = np.average(f1_per_class, weights=np.bincount(all_targets))

print(f"Micro F1 Score (manual): {micro_f1_manual}")
print(f"Macro F1 Score (manual): {macro_f1_manual}")
print(f"Weighted F1 Score (manual): {weighted_f1_manual}")


