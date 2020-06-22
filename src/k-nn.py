from metrics import Metrics

import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
from sklearn import neighbors

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

knn_model = neighbors.KNeighborsClassifier(n_jobs = -1)

start = time.time()
knn_model.fit(Xtr_f, ytr)
end = time.time()

print(f"[features] Training time: {end-start}s")

#prediction on training set
start = time.time()
knn_pred_training = knn_model.predict(Xtr_f)
end = time.time()
print(f"[features] K-NN prediction on training set time: {end-start}s")

#prediction on validation set
start = time.time()
knn_pred_val = knn_model.predict(Xvl_f)
end = time.time()
print(f"[features] K-NN prediction on training set time: {end-start}s")

# Cast the predictions into Pytorch tensors
knn_pred_training = torch.tensor(knn_pred_training)
knn_pred_val = torch.tensor(knn_pred_val)

# print performances
print(f"\n[features] Performance on training set: {metrics.evaluate(ytr, knn_pred_training)}")
print(f"[features] Performance on validation set: {metrics.evaluate(yvl, knn_pred_val)}")

# We can try to modify the number of neighbors (n_neighbors)
knn_models = [neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                             n_jobs=-1).fit(Xtr_f, ytr)
             for n_neighbors in [3,5,7,9,11]
             ]

#validation predictions
knn_models_pred_val = [model.predict(Xvl_f) for model in knn_models]

#calculate performances
knn_models_performance_val = [metrics.evaluate(yvl, yp) for yp in knn_models_pred_val]

best_model_idx = torch.tensor([res['Atot'] for res in knn_models_performance_val]).argmax()

best_model_f = knn_models[best_model_idx]
best_model_pred_training = best_model_f.predict(Xtr_f)

print(f"\n[features] Best model parameters: n_neighbors={best_model_f.n_neighbors}")
print(f"[features] Best model performances on validation set: \n\t{knn_models_performance_val[best_model_idx]}")
print(f"[features] Best model performances on training set: \n\t{metrics.evaluate(ytr, best_model_pred_training)}")

torch.save(best_model_f, 'models/knn_features.pt')

#===================== IMAGES ===========================

knn_model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs = -1)

start = time.time()
knn_model.fit(Xtr_im, ytr)
end = time.time()

print(f"[images] Training time: {end-start}s")

#prediction on training set
start = time.time()
knn_pred_training = knn_model.predict(Xtr_im)
end = time.time()

print(f"[images] K-NN prediction on training set time: {end-start}s") #440s

#prediction on validation set
start = time.time()
knn_pred_val = knn_model.predict(Xvl_im)
end = time.time()

print(f"[images] K-NN prediction on validation set time: {end-start}s") #145s

# Cast the predictions into Pytorch tensors
knn_pred_training = torch.tensor(knn_pred_training)
knn_pred_val = torch.tensor(knn_pred_val)

# print performances
print(f"\n[images] Performance on training set: {metrics.evaluate(ytr, knn_pred_training)}")
print(f"[images] Performance on validation set: {metrics.evaluate(yvl, knn_pred_val)}")

torch.save(knn_model, 'models/knn_images.pt')

'''

model = torch.load('models/knn_features.pt')

pred = model.predict(Xvl_f)

print(metrics.evaluate(yvl,pred))
print(model.n_neighbors)
'''