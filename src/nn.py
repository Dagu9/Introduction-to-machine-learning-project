import torch 
import torch.utils.data as data
from metrics import Metrics
import torch.nn as nn
import torch.utils.tensorboard as tb
import torch.optim as optim
from fullyConnectedNN import FullyConnectedNN
from convNet import ConvNet
from import_data import getLabelName
import sys

def normalize(X):
        mean = X.mean()
        std_dev = X.std()

        X = X-mean
        X = X/std_dev
        
        return X

def train_one_epoch(model, loss_func, metric_tracker, dataloader, optimizer, epoch, tblog=None):

    # set model to training mode
    model.train()

    # clear confusion matrix
    metric_tracker.clear()

    for i, (X, yt) in enumerate(dataloader):
        X = X.to("cuda:0")
        yt = yt.to("cuda:0")

        # set gradient to zero
        optimizer.zero_grad()

        # predictions
        Y = model(X)
        # compute loss
        loss = loss_func(Y, yt.long())

        # predicted class label
        y = Y.argmax(-1)

        # add predicted class to confusion matrix
        yt = yt.to("cpu")
        y = y.to("cpu")
        metric_tracker.add(y, yt)

        if tblog:
            tblog.add_scalar('train/loss', loss.item(), epoch*len(dataloader)+i)
        
        # compute gradient
        loss.backward()
        # update parameters
        optimizer.step()

def validate(model, metric_tracker, dataloader):
    # set model to evaluation mode
    model.eval()

    # clear confusion matrix
    metric_tracker.clear()

    with torch.no_grad():
        for (X, yt) in dataloader:
            X = X.to("cuda:0")
            yt = yt.to("cuda:0")

            # predictions
            Y = model(X)

            Y = Y.to("cpu")
            yt = yt.to("cpu")

            # predicted label
            y = Y.argmax(-1)

            metric_tracker.add(y, yt)

def train(model, trDataloader, vlDataloader, optimizer, lr_scheduler, num_epochs, tblog=None):

    # create loss function used for training
    loss_func = nn.CrossEntropyLoss()

    # create metric tracker
    metric_tracker = Metrics(model.num_classes)

    for epoch in range(1, num_epochs+1):

        print(f"-- EPOCH {epoch}/{num_epochs} ----------------------\n")

        # train model
        train_one_epoch(model, loss_func, metric_tracker, trDataloader, optimizer, epoch, tblog)

        print(f"\tTRAIN | acc:{metric_tracker.acc()} | atot:{metric_tracker.accTot()}")

        # track metrics in tensorboard
        if tblog:
            tblog.add_scalar('train/atot', metric_tracker.accTot(), epoch)
            tblog.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

        # validate model
        validate(model, metric_tracker, vlDataloader)

        print(f"\tEVAL | acc:{metric_tracker.acc()} | atot:{metric_tracker.accTot()}\n")

        if tblog:
            tblog.add_scalar('val/atot', metric_tracker.accTot(), epoch)

        if metric_tracker.accTot()>0.78:
            torch.save(conv_net, 'models/cnn-4c-5fc-ReLu.pt')
        
        lr_scheduler.step()

def run_network(name, model, trDataloader, vlDataloader, num_epochs=25):

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15,20,25], gamma=0.5)

    tblog = tb.SummaryWriter(f"networks/{name}")

    train(model, trDataloader, vlDataloader, optimizer, lr_scheduler, num_epochs=num_epochs, tblog=tblog)


print("[*] Loading data...")
Xtr_f = torch.load('tensors/Xtr_f.pt')
Xtr_im = torch.load('tensors/Xtr_im.pt')
Xtr_im_f = torch.load('tensors/Xtr_im_f.pt')
ytr = torch.load('tensors/ytr.pt')
Xvl_f = torch.load('tensors/Xvl_f.pt')
Xvl_im = torch.load('tensors/Xvl_im.pt')
Xvl_im_f = torch.load('tensors/Xvl_im_f.pt')
yvl = torch.load('tensors/yvl.pt')
#Xts_f = torch.load('tensors/Xts_f.pt')
#Xts_im = torch.load('tensors/Xts_im.pt')
#Xts_im_f = torch.load('tensors/Xts_im_f.pt')
print("[*] Data loaded!")

# normalization
Xtr_f = normalize(Xtr_f)
Xtr_im = normalize(Xtr_im)
Xtr_im_f = normalize(Xtr_im_f)
Xvl_f = normalize(Xvl_f)
Xvl_im = normalize(Xvl_im)
Xvl_im_f = normalize(Xvl_im_f)
#Xts_f = normalize(Xts_f)
#Xts_im = normalize(Xts_im)
#Xts_im_f = normalize(Xts_im_f)

BATCH_SIZE = 128

# create PyTorch Datasets
trDataset_f = data.TensorDataset(Xtr_f, ytr)
vlDataset_f = data.TensorDataset(Xvl_f, yvl)
trDataset_im = data.TensorDataset(Xtr_im, ytr)
vlDataset_im = data.TensorDataset(Xvl_im, yvl)
#trDataset_im_f = data.TensorDataset(Xtr_im_f, ytr)
#vlDataset_im_f = data.TensorDataset(Xvl_im_f, yvl)

# create dataloaders
trDataloader_f = data.DataLoader(trDataset_f, batch_size=BATCH_SIZE, shuffle=False)
vlDataloader_f = data.DataLoader(vlDataset_f, batch_size=BATCH_SIZE, shuffle=False)
trDataloader_im = data.DataLoader(trDataset_im, batch_size=BATCH_SIZE, shuffle=True)
vlDataloader_im = data.DataLoader(vlDataset_im, batch_size=BATCH_SIZE, shuffle=True)
#trDataloader_im_f = data.DataLoader(trDataset_im_f, batch_size=BATCH_SIZE)
#vlDataloader_im_f = data.DataLoader(vlDataset_im_f, batch_size=BATCH_SIZE)
'''
#===================== FEATURES ===========================
# create network
fully_connected_features = FullyConnectedNN([84,64,32,16,8], num_classes=4, activation_type=nn.ReLU)

# run training
name = 'features-FC-64-32-16-8-ReLU'
run_network(name, fully_connected_features, trDataloader_f, vlDataloader_f, num_epochs=200)

torch.save(fully_connected_features, 'models/nn-fc-64-32-16-8-ReLU.pt')
'''
#===================== IMAGES ===========================
#fully_connected_images = FullyConnectedNN([256**2, 10000, 512, 256, 128, 64], num_classes=4, activation_type=nn.Sigmoid)
#name = 'images-FC-512-256-128-64-ReLU'
#run_network(name, fully_connected_images, trDataloader_im, vlDataloader_im, num_epochs=25)

conv_net = ConvNet(num_classes=4)
conv_net.to(torch.device("cuda:0"))
name = 'images-5-conv-6-fc-ReLU'
run_network(name, conv_net, trDataloader_im, vlDataloader_im, num_epochs=35)

#storch.save(conv_net, 'models/cnn-4c-5fc-ReLu.pt')


'''
net = torch.load('models/nn-fc-64-32-16-8-ReLU.pt')
results = {}
for i,pred in enumerate(net.forward(Xts_f)):
    label = pred.argmax(-1).item()
    results[str(i).rjust(4,'0')] = getLabelName(label)

with open('predictions_nn_features','w') as f:
    for i,im in enumerate(results.keys()):
        if(i!=1190):
            f.write(im+" "+results[im]+"\n")
        else:
            f.write(im+" "+results[im])

'''