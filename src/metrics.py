import torch

class Metrics():
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.C = torch.zeros(num_classes, num_classes)

    def add(self, yp, yt):
        # compute the confusion matrix C, where C_ij represents the number of images 
        # of class i classified with class j
        with torch.no_grad():
            self.C += (yt*self.num_classes+yp).bincount(minlength=self.num_classes**2).view(self.num_classes,self.num_classes).float()

    def acc(self):
        self.ac = torch.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            # true positive
            tp = self.C.diag()[i].item()
            # false negative + true positive = row i of the confusion matrix
            tp_fp = self.C[i].sum().item()
            # avoid division by zero
            tp_fp = max(1, tp_fp)
            
            self.ac[i] = tp/tp_fp
        
        return self.ac
    
    def accTot(self):
        self.atot = self.ac.sum()/self.num_classes
        return self.atot

    def clear(self):
        self.C.zero_()
        
    def evaluate(self, yt, yp):
        # yt: 1D tensor of size n containing the target class labels
        # yp: 1D tensor of size n containing the predicted class labels
        # numclasses: total number of classes
        
        self.clear()
        self.add(yp,yt)

        ac = self.acc()
        atot = self.accTot()

        return {'Ac':ac, 
                'Atot': atot.item() } 