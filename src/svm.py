from metrics import Metrics

import torch
import time
from sklearn import svm
import numpy as np

metrics = Metrics()

print("[*] Loading data...")
Xtr_f = torch.load('tensors/Xtr_f.pt')
Xtr_im = torch.load('tensors/Xtr_im.pt')
Xtr_im_f = torch.load('tensors/Xtr_im_f.pt')
ytr = torch.load('tensors/ytr.pt')
Xvl_f = torch.load('tensors/Xvl_f.pt')
Xvl_im = torch.load('tensors/Xvl_im.pt')
Xvl_im_f = torch.load('tensors/Xvl_im_f.pt')
yvl = torch.load('tensors/yvl.pt')
Xts_f = torch.load('tensors/Xts_f.pt')
Xts_im = torch.load('tensors/Xts_im.pt')
Xts_im_f = torch.load('tensors/Xts_im_f.pt')
print("[*] Data loaded!")

#===================== FEATURES ===========================

svm_model = svm.SVC(gamma='scale')

start = time.time()
svm_model.fit(Xtr_f,ytr)
end = time.time()

print(f"[features] Training time: {end-start}s")

# prediction and evaluation

start = time.time()
svm_pred_training = svm_model.predict(Xtr_f)
end = time.time()
print(f"[features] Prediction on training set time: {end-start}")

start = time.time()
svm_pred_val = svm_model.predict(Xvl_f)
end = time.time()
print(f"[features] Prediction on validation set time: {end-start}")

print(f"\n[features] Performance on training set: {metrics.evaluate(ytr, svm_pred_training)}") #0.89
print(f"[features] Performance on validation set: {metrics.evaluate(yvl, svm_pred_val)}")   #0.72

# We can try to modify the regularization parameter C and the kernel 

svm_models = [svm.SVC(C=c, kernel=kernel, gamma='scale').fit(Xtr_f, ytr)
                for c in range(1,10,1)
                for kernel in ['linear','rbf','poly','sigmoid']
             ]

# predictions on validation set
svm_models_pred_val = [model.predict(Xvl_f) for model in svm_models]

# performances
svm_models_perf_val = [metrics.evaluate(yvl, yp) for yp in svm_models_pred_val]

best_idx = torch.tensor([res['Atot'] for res in svm_models_perf_val]).argmax()

best_svm_model_f = svm_models[best_idx]

print(f"\n[features] Best model parameters: C={best_svm_model_f.C}, kernel={best_svm_model_f.kernel}") 
print(f"[features] Performance on validation set: {svm_models_perf_val[best_idx]}")   #0.73
print(f"[features] Performance on training set: {metrics.evaluate(ytr, best_svm_model_f.predict(Xtr_f))}") 

torch.save(best_svm_model_f, 'models/svm_features.pt')

#===================== IMAGES ===========================

svm_model = svm.LinearSVC(C=7)

start = time.time()
svm_model.fit(Xtr_im,ytr)
end = time.time()

print(f"[images] Training time: {end-start}s")  # 1800 s

# prediction and evaluation

start = time.time()
svm_pred_training = svm_model.predict(Xtr_im)
end = time.time()
print(f"[images] Prediction on training set time: {end-start}")   

start = time.time()
svm_pred_val = svm_model.predict(Xvl_im)
end = time.time()
print(f"[images] Prediction on validation set time: {end-start}")

svm_pred_training = torch.tensor(svm_pred_training)
svm_pred_val = torch.tensor(svm_pred_val)

print(f"\n[images] Performance on training set: {metrics.evaluate(ytr, svm_pred_training)}")  # 1
print(f"[images] Performance on validation set: {metrics.evaluate(yvl, svm_pred_val)}")      # 0.67

torch.save(svm_model, 'models/svm_images.pt')
'''
model_f = torch.load('models/svm_features.pt')


pred = model_f.predict(Xvl_f)
print(f"{metrics.evaluate(yvl, pred)}, {model_f.C}, {model_f.kernel}")
'''