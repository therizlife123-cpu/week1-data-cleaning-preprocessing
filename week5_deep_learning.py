# Week 5 — Deep Learning Application in Data Science
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

torch.manual_seed(42)
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

model = torch.nn.Sequential(
    torch.nn.Linear(64,128), torch.nn.ReLU(), torch.nn.Dropout(0.25),
    torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Dropout(0.20),
    torch.nn.Linear(64,10)
)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

rng=np.random.default_rng(42); idx=np.arange(len(X_train)); rng.shuffle(idx)
cut=int(0.8*len(idx)); tr_idx, val_idx=idx[:cut],idx[cut:]
Xt=torch.tensor(X_train); yt=torch.tensor(y_train,dtype=torch.long)
best_state=None; best_val=float('inf'); wait=0
for epoch in range(100):
    model.train(); optimizer.zero_grad()
    loss=criterion(model(Xt[tr_idx]),yt[tr_idx]); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): val_loss=criterion(model(Xt[val_idx]),yt[val_idx]).item()
    if val_loss < best_val-1e-4:
        best_val=val_loss; best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
    else: wait += 1
    if wait >= 10: break

model.load_state_dict(best_state); model.eval()
with torch.no_grad(): predictions=model(torch.tensor(X_test)).argmax(1).numpy()
print('Test accuracy:',accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
