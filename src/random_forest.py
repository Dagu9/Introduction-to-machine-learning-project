from import_data import Import_data
from metrics import Metrics

import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import ensemble
import time

dataloader = Import_data()
metrics = Metrics()

print("[*] Loading data...")
Xtr_f, Xtr_im, Xtr_im_f, ytr, Xvl_f, Xvl_im, Xvl_im_f, yvl, Xts_f, Xts_im, Xts_im_f = dataloader.get_shuffled_data()
torch.save(Xtr_f, 'tensors/Xtr_f.pt')
torch.save(Xtr_im, 'tensors/Xtr_im.pt')
torch.save(Xtr_im_f, 'tensors/Xtr_im_f.pt')
torch.save(ytr, 'tensors/ytr.pt')
torch.save(Xvl_f, 'tensors/Xvl_f.pt')
torch.save(Xvl_im, 'tensors/Xvl_im.pt')
torch.save(Xvl_im_f, 'tensors/Xvl_im_f.pt')
torch.save(yvl, 'tensors/yvl.pt')
torch.save(Xts_f, 'tensors/Xts_f.pt')
torch.save(Xts_im, 'tensors/Xts_im.pt')
torch.save(Xts_im_f, 'tensors/Xts_im_f.pt')
'''
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
'''
print("[*] Data loaded!")
#dataloader.show_images()

#===================== FEATURES ===========================
# classifier with 100 trees and the gini criterion
rf_model = ensemble.RandomForestClassifier(n_estimators=100)

# fit the model with features
start = time.time()
rf_model.fit(Xtr_f, ytr)
end = time.time()
print(f"[features] Random Forest (gini, 100 trees) training time: {end-start}s")

# Prediction on training set
start = time.time()
rf_pred_training = rf_model.predict(Xtr_f)
end = time.time()
print(f"[features] Random Forest (gini, 100 trees) prediction on training set time: {end-start}s")

# Prediction on validation set
start = time.time()
rf_pred_val = rf_model.predict(Xvl_f)
end = time.time()
print(f"[features] Random Forest (gini, 100 trees) prediction on validation set time: {end-start}s")

# Cast the predictions into Pytorch tensors
rf_pred_training = torch.tensor(rf_pred_training)
rf_pred_val = torch.tensor(rf_pred_val)

# print performances
print(f"\n[features] Performance on training set: {metrics.evaluate(ytr, rf_pred_training)}")
print(f"[features] Performance on validation set: {metrics.evaluate(yvl, rf_pred_val)}")

# try to train different models with 2, 5, 10 as the minimum number of samples that we have in a leaf and 
# with criterion "gini" and "entropy". 
# We use 20, 40, 60, 80, 100 as the number of trees
forests = [ensemble.RandomForestClassifier(n_estimators=n_trees, 
                                           criterion=criterion, 
                                           min_samples_leaf=min_samples_leaf, 
                                           n_jobs=-1).fit(Xtr_f,ytr)
           for n_trees in [20,40,60,80,100]
           for criterion in ["gini","entropy"]
           for min_samples_leaf in [2,5,10]
          ]

# compute predictions
forests_pred_val = [forest.predict(Xvl_f) for forest in forests]

# compute performances
forests_performance_vl = [metrics.evaluate(yvl, yp) for yp in forests_pred_val]

# select the best model based on total accuracy
best_rf_idx = torch.tensor([res['Atot'] for res in forests_performance_vl]).argmax()

best_model = forests[best_rf_idx]

# print accuracies and parameters of the best model
print(f"\n[features] Best model parameters: criterion={best_model.criterion} n_estimators={best_model.n_estimators} min_samples_leaf={best_model.min_samples_leaf}")
print(f"[features] Best model performances on validation set: \n\tclasses: {forests_performance_vl[best_rf_idx]['Ac']} \n\ttotal: {forests_performance_vl[best_rf_idx]['Atot']}")

torch.save(best_model, 'models/rf_features.pt')

#===================== IMAGES ===========================

# classifier with 100 trees and the gini criterion
rf_model_im = ensemble.RandomForestClassifier(n_estimators=100)

# fit the model with images
start = time.time()
rf_model_im.fit(Xtr_im, ytr)
end = time.time()

print(f"\n[images] Random Forest (gini, 100 trees) training time: {end-start}s")

# Prediction on training set
start = time.time()
rf_pred_training = rf_model_im.predict(Xtr_im)
end = time.time()

print(f"[images] Random Forest (gini, 100 trees) prediction on training set time: {end-start}s")

# Prediction on validation set
start = time.time()
rf_pred_val = rf_model_im.predict(Xvl_im)
end = time.time()

print(f"[images] Random Forest (gini, 100 trees) prediction on validation set time: {end-start}s")

# Cast the predictions into Pytorch tensors
rf_pred_training = torch.tensor(rf_pred_training)
rf_pred_val = torch.tensor(rf_pred_val)

# print performances
print(f"\n[images] Performance on training set: {metrics.evaluate(ytr, rf_pred_training)}")
print(f"[images] Performance on validation set: {metrics.evaluate(yvl, rf_pred_val)}")

# We could try different models as with features but the training takes a lot of time so we can try 5, 10 as the minimum number of samples that we have in a leaf and 
# 40, 60, 80, 100 as the number of trees. 
# We try with both "gini" and "entropy" criterion.
forests = [ensemble.RandomForestClassifier(n_estimators=n_trees, 
                                           criterion=criterion, 
                                           min_samples_leaf=min_samples_leaf, 
                                           n_jobs=-1).fit(Xtr_im,ytr)
           for n_trees in [40,60,80,100]
           for criterion in ["gini","entropy"]
           for min_samples_leaf in [5,10]
          ]

# compute predictions
forests_pred_val = [forest.predict(Xvl_im) for forest in forests]

# compute performances
forests_performance_vl = [metrics.evaluate(yvl, yp) for yp in forests_pred_val]

# select the best model based on total accuracy
best_rf_idx = torch.tensor([res['Atot'] for res in forests_performance_vl]).argmax()

best_model_im = forests[best_rf_idx]

# print accuracies and parameters of the best model
print(f"\n[images] Best model parameters: criterion={best_model_im.criterion} n_estimators={best_model_im.n_estimators} min_samples_leaf={best_model_im.min_samples_leaf}")
print(f"[images] Best model performances on validation set: \n\tclasses: {forests_performance_vl[best_rf_idx]['Ac']} \n\ttotal: {forests_performance_vl[best_rf_idx]['Atot']}")

# We can try to adjust weights and see if we can do better.
forests = [ensemble.RandomForestClassifier(n_estimators=best_model_im.n_estimators,
                                           criterion=best_model_im.criterion,
                                           min_samples_leaf=best_model_im.min_samples_leaf,
                                           class_weight={0:w1, 1:w2, 2:w3, 3:w4},
                                           n_jobs=-1).fit(Xtr_im, ytr)
           for w1 in [1,5,10]
           for w2 in [1,5,10]
           for w3 in [1,5,10]
           for w4 in [1,5,10]
          ]

# compute predictions
forests_pred_val = [forest.predict(Xvl_im) for forest in forests]

# compute performances
forests_performance_vl = [metrics.evaluate(yvl, yp) for yp in forests_pred_val]

# select the best model based on total accuracy
best_rf_idx = torch.tensor([res['Atot'] for res in forests_performance_vl]).argmax()

best_model_im = forests[best_rf_idx]

# print accuracies and parameters of the best model
print(f"\n[images] Best model parameters: class_weight={best_model_im.class_weight}")
print(f"[images] Best model performances on validation set: \n\tclasses: {forests_performance_vl[best_rf_idx]['Ac']} \n\ttotal: {forests_performance_vl[best_rf_idx]['Atot']}")

# Check if model is overfitted
best_model_pred_training = best_model_im.predict(Xtr_im)

print(f"[images] Best model performances on training set:\n{metrics.evaluate(ytr, best_model_pred_training)}")

# We try to increase min_samples_leaf to avoid overfitting
new_best_model_im = ensemble.RandomForestClassifier(n_estimators=best_model_im.n_estimators,
                                           criterion=best_model_im.criterion,
                                           min_samples_leaf=9,
                                           class_weight=best_model_im.class_weight,
                                           n_jobs=-1).fit(Xtr_im, ytr)

new_best_model_pred_training = new_best_model_im.predict(Xtr_im)
new_best_model_pred_val = new_best_model_im.predict(Xvl_im)

print(f"\n[images] New best model performances on training set:\n\t{metrics.evaluate(ytr, new_best_model_pred_training)}")
print(f"[images] New best model performances on validation set:\n\t{metrics.evaluate(yvl, new_best_model_pred_val)}")

torch.save(new_best_model_im, 'models/rf_images.pt')

#=========================== IMAGES + FEATURES =======================

# classifier with 100 trees and the gini criterion
rf_model = ensemble.RandomForestClassifier(n_estimators=100)

# fit the model with images
start = time.time()
rf_model.fit(Xtr_im_f, ytr)
end = time.time()
print(f"\n[images+features] Random Forest (gini, 100 trees) training time: {end-start}s")

# Prediction on training set
start = time.time()
rf_pred_training = rf_model.predict(Xtr_im_f)
end = time.time()
print(f"[images+features] Random Forest (gini, 100 trees) prediction on training set time: {end-start}s")

# Prediction on validation set
start = time.time()
rf_pred_val = rf_model.predict(Xvl_im_f)
end = time.time()
print(f"[images+features] Random Forest (gini, 100 trees) prediction on validation set time: {end-start}s")

# Cast the predictions into Pytorch tensors
rf_pred_training = torch.tensor(rf_pred_training)
rf_pred_val = torch.tensor(rf_pred_val)

# print performances
print(f"\n[images+features] Performance on training set: {metrics.evaluate(ytr, rf_pred_training)}")
print(f"[images+features] Performance on validation set: {metrics.evaluate(yvl, rf_pred_val)}")

forests = [ensemble.RandomForestClassifier(n_estimators=n_trees, 
                                           criterion=criterion, 
                                           min_samples_leaf=min_samples_leaf, 
                                           n_jobs=-1).fit(Xtr_im_f,ytr)
           for n_trees in [40,60,80,100]
           for criterion in ["gini","entropy"]
           for min_samples_leaf in [5,10]
          ]

# compute predictions
forests_pred_val = [forest.predict(Xvl_im_f) for forest in forests]

# compute performances
forests_performance_vl = [metrics.evaluate(yvl, yp) for yp in forests_pred_val]

# select the best model based on total accuracy
best_rf_idx = torch.tensor([res['Atot'] for res in forests_performance_vl]).argmax()

best_model_im_f = forests[best_rf_idx]

# print accuracies and parameters of the best model
print(f"\n[images+features] Best model parameters: criterion={best_model_im_f.criterion} n_estimators={best_model_im_f.n_estimators} min_samples_leaf={best_model_im_f.min_samples_leaf}")
print(f"[images+features] Best model performances on validation set: \n\tclasses: {forests_performance_vl[best_rf_idx]['Ac']} \n\ttotal: {forests_performance_vl[best_rf_idx]['Atot']}")

# We can try to adjust weights and see if we can do better.
forests = [ensemble.RandomForestClassifier(n_estimators=best_model_im_f.n_estimators,
                                           criterion=best_model_im_f.criterion,
                                           min_samples_leaf=best_model_im_f.min_samples_leaf,
                                           class_weight={0:w1, 1:w2, 2:w3, 3:w4},
                                           n_jobs=-1).fit(Xtr_im_f, ytr)
           for w1 in [1,5,10]
           for w2 in [1,5,10]
           for w3 in [1,5,10]
           for w4 in [1,5,10]
          ]

# compute predictions
forests_pred_val = [forest.predict(Xvl_im_f) for forest in forests]

# compute performances
forests_performance_vl = [metrics.evaluate(yvl, yp) for yp in forests_pred_val]

# select the best model based on total accuracy
best_rf_idx = torch.tensor([res['Atot'] for res in forests_performance_vl]).argmax()

new_best_model_im_f = forests[best_rf_idx]

# print accuracies and parameters of the best model
print(f"\n[images+features]  class_weight={new_best_model_im_f.class_weight}")
print(f"[images+features] Best model performances on validation set: \n\tclasses: {forests_performance_vl[best_rf_idx]['Ac']} \n\ttotal: {forests_performance_vl[best_rf_idx]['Atot']}")

torch.save(new_best_model_im_f, 'models/rf_images_features.pt')


'''
rf_f_model = torch.load('models/rf_features.pt')
rf_im_model = torch.load('models/rf_images.pt')
rf_im_f_model = torch.load('models/rf_images_features.pt')

pred_f = rf_f_model.predict(Xvl_f)
pred_im = rf_im_model.predict(Xvl_im)
pred_im_f = rf_im_f_model.predict(Xvl_im_f)

print(f"[f] {metrics.evaluate(yvl, pred_f)}") #0.7322909832000732
print(f"[f] trees:{rf_f_model.n_estimators}, criterion:{rf_f_model.criterion}, min_leaf:{rf_f_model.min_samples_leaf}")
print(f"[im] {metrics.evaluate(yvl, pred_im)}") #0.7227548956871033
print(f"[im] trees:{rf_im_model.n_estimators}, criterion:{rf_im_model.criterion}, min_leaf:{rf_im_model.min_samples_leaf}")
print(f"[im_f] {metrics.evaluate(yvl, pred_im_f)}") #0.7558963894844055
print(f"[im_f] trees:{rf_im_f_model.n_estimators}, criterion:{rf_im_f_model.criterion}, min_leaf:{rf_im_f_model.min_samples_leaf}")
'''