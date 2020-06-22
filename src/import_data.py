import numpy as np 
import torch
from PIL import Image
import matplotlib.pyplot as plt

class Import_data:
    def __init__(self):
        

        self.features_folder = '../project_data/features/'
        self.images_folder = '../project_data/images/'
        self.labels_folder = '../project_data/labels/'

        self.LEN_TRAIN = 3569
        self.LEN_VAL = 1189
        self.LEN_TEST = 1191

        self.Xtr_f = []
        self.Xvl_f = []
        self.Xts_f = []

        self.Xtr_im = torch.zeros((self.LEN_TRAIN, 256, 256))
        self.ytr = torch.zeros(self.LEN_TRAIN, dtype=torch.int32)

        self.Xvl_im = torch.zeros((self.LEN_VAL, 256, 256))
        self.yvl = torch.zeros(self.LEN_VAL, dtype=torch.int32)

        self.Xts_im = torch.zeros((self.LEN_TEST,256,256))

        self.labels = {
            'bacteria':0,
            'corona':1,
            'normal':2,
            'viral':3
        }

    def load_features(self):
        # read npy files for training set
        for i in range(self.LEN_TRAIN):
            features_arr = np.load('{}train/{:04d}.npy'.format(self.features_folder, i))
            self.Xtr_f.append(features_arr)

        # read npy files for validation set
        for i in range(self.LEN_VAL):
            features_arr = np.load('{}val/{:04d}.npy'.format(self.features_folder, i))
            self.Xvl_f.append(features_arr)

        # read npy files for test set
        for i in range(self.LEN_TEST):
            features_arr = np.load('{}test/{:04d}.npy'.format(self.features_folder, i))
            self.Xts_f.append(features_arr)

        # create torch tensors from lists
        self.Xtr_f = torch.tensor(self.Xtr_f)
        self.Xvl_f = torch.tensor(self.Xvl_f)
        self.Xts_f = torch.tensor(self.Xts_f)

        # shape of features tensors
        #print(f"Training features shape: {Xtr_f.shape}")
        #print(f"Validation features shape: {Xvl_f.shape}")
        #print(f"Test features shape: {Xts_f.shape}")

        return self.Xtr_f, self.Xvl_f, self.Xts_f

    def load_images(self):
        import os

        # load images from directories
        for label in self.labels.keys():
            # training set
            files = [name for name in os.listdir(f"{self.images_folder}/train/{label}/")]
            for im in files:
                image = Image.open(f"{self.images_folder}/train/{label}/{im}").convert('L')
                # get index of image
                idx = int(im[:4])
                self.Xtr_im[idx] = torch.from_numpy(np.asarray(image))
                self.ytr[idx] = self.labels[label]
            
            # validation set
            files = [name for name in os.listdir(f"{self.images_folder}/val/{label}/")]
            for im in files:
                image = Image.open(f"{self.images_folder}/val/{label}/{im}").convert('L')
                idx = int(im[:4])
                self.Xvl_im[idx] = torch.from_numpy(np.asarray(image))
                self.yvl[idx] = self.labels[label]

        # test set
        files = [name for name in os.listdir(f"{self.images_folder}/test/no_labels/")]
        for im in files:
            image = Image.open(f"{self.images_folder}/test/no_labels/{im}").convert('L')
            idx = int(im[:4])
            self.Xts_im[idx] = torch.from_numpy(np.asarray(image))

        #print(f"Training images shape: {self.Xtr_im.shape},  training labels shape: {self.ytr.shape}")
        #print(f"Validation images shape: {self.Xvl_im.shape}, validation labels shape: {self.yvl.shape}")
        #print(f"Test images shape: {self.Xts_im.shape}")

        #reshape
        self.Xtr_im = self.Xtr_im.view(self.Xtr_im.shape[0], -1)
        self.Xvl_im = self.Xvl_im.view(self.Xvl_im.shape[0], -1)
        self.Xts_im = self.Xts_im.view(self.Xts_im.shape[0], -1)

        return self.Xtr_im, self.ytr, self.Xvl_im, self.yvl, self.Xts_im

    def get_shuffled_data(self):
        self.load_features()
        self.load_images()

        shuffled_indices = torch.randperm(self.Xtr_im.shape[0])

        self.Xtr_f = self.Xtr_f[shuffled_indices]
        self.Xtr_im = self.Xtr_im[shuffled_indices]
        self.ytr = self.ytr[shuffled_indices]

        # validation set
        shuffled_indices = torch.randperm(self.Xvl_im.shape[0])

        self.Xvl_f = self.Xvl_f[shuffled_indices]
        self.Xvl_im = self.Xvl_im[shuffled_indices]
        self.yvl = self.yvl[shuffled_indices]

        Xtr_im_f = torch.cat((self.Xtr_im, self.Xtr_f),1)
        Xvl_im_f = torch.cat((self.Xvl_im, self.Xvl_f),1)
        Xts_im_f = torch.cat((self.Xts_im, self.Xts_f),1)

        return self.Xtr_f, self.Xtr_im, Xtr_im_f, self.ytr, self.Xvl_f, self.Xvl_im, Xvl_im_f, self.yvl, self.Xts_f, self.Xts_im, Xts_im_f

    def show_images(self):
        
        # find one image per label
        indices = []
        for label in self.labels.values():
            im_index = (self.ytr==label).nonzero()[0].item()
            indices.append(im_index)

        # plot images
        for i, label in enumerate(self.labels.keys()):
            ax = plt.subplot(1, 4, i+1)
            ax.axis('off')
            plt.imshow(self.Xtr_im[indices[i]].view(256,256), cmap='gray')
            plt.title(f"{label}")
            
        plt.show()


def getLabelName(label):
        if label == 0:
            return 'Bacterial'
        elif label == 1:
            return 'COVID-19'
        elif label == 2:
            return 'Normal'
        elif label == 3:
            return 'Viral'