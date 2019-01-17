from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

import json


from PIL import Image, ImageOps

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


class GuidedBackprop:
    reluList = []
    datatransform = None
    md = None
    labels_dict = None
    maxBack = []

    def __init__(self,md):


        # Add model hooks
        self.hookModel(md)

        # Define transform operations
        self.data_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Class names
        self.labels_dict = json.load(open("labels_name.json", 'r'))

    def hookModel(self,md):
        reluList = self.reluList

        def reluForw(mod, x_in, x_out):
            self.reluList.append(x_out)

        def reluBack(mod, x_in, x_out):
            # Save Max Activation layer
            reluCopy = self.reluList[-1]
            # Max activation
            #mxIdx = np.where(x_in[0] == x_in[0].max())

            # Max mean activation
            # reluSum = np.sum(x_in[0].detach().numpy()[0],axis=(1,2))
            # mxIdx = [0,np.where(reluSum == reluSum.max())[0]]

            # Perc 90
            pN = 0.9
            sortd_relu = [sorted(el.detach().numpy().reshape(-1)) for el in x_in[0][0]]
            perc_relu = list(map(lambda x: x[int(0.9 * len(x))], sortd_relu))
            sortd_list = sorted(zip(perc_relu, reluCopy[0].detach().numpy()), key=lambda  x: x[0], reverse=True)

            #mxActiv = reluCopy[0, mxIdx[1]].detach().numpy()[0]
            mxActiv = sortd_list[0][1]
            #self.maxBack.append(mxActiv)
            #self.maxBack.append(list(map(lambda x: x[1],sortd_list)))
            self.maxBack.append(sortd_list)


            # Hook logic
            last_mask = ((self.reluList.pop() > 0) & (x_in[0] > 0)).float()
            mod_x_out = last_mask * x_in[0]

            return (mod_x_out,)

        # Register hooks
        # First ReLU
        md.relu.register_forward_hook(reluForw)
        md.relu.register_backward_hook(reluBack)

        # layers ReLU
        idx = 5
        for i in map(str, range(1, idx)):
            md._modules['layer' + i]._modules['0'].relu.register_forward_hook(reluForw)
            md._modules['layer' + i]._modules['1'].relu.register_backward_hook(reluBack)

        self.md = md

    def getGB(self,im,desired_class=None):
        # Make sure eval mode
        self.md.eval()
        # Erase current grad
        self.md.zero_grad()

        ###
        #
        # Transform image
        #
        ###

        t_dataset = self.data_transforms(im)
        t_dataset = t_dataset.reshape([1] + list(t_dataset.shape))

        # Keep a copy
        out_original = t_dataset.clone()

        # Enable autograd
        t_dataset.requires_grad_(True)

        ###
        #
        # Prediction
        #
        ###

        pred = self.md(t_dataset)

        # Print prediction
        bstIdx = np.where(pred == pred.max())[1][0]
        print("{} {} ({:.3f}/{:.3f})".format(
            self.labels_dict[str(np.where(pred == pred.max())[1][0])],
            str(bstIdx),
            pred.max(),
            pred[0,desired_class if desired_class is not None else bstIdx]))

        # Build gradient init
        v = np.zeros((1, 1000))
        if desired_class is None:
            v[0, np.where(pred == pred.max())[1][0]] = 1
        else:
            v[0, desired_class] = 1
        v = torch.Tensor(v)

        ###
        #
        # Save Max Activation forward
        #
        ###
        max_activ = list(map(lambda x: (x.max(), np.where(x==x.max())),self.reluList))
        max_activation_maps = [reEl[0,maxEl[1][1],...].detach().numpy()[0] for reEl, maxEl in zip(self.reluList,max_activ)]

        ###
        #
        # Backpropagation
        #
        ###
        pred.backward(gradient=v)

        out_img = t_dataset.grad.numpy()[0].transpose((1,2,0))
        out_original = out_original.numpy()[0].transpose((1,2,0))

        return out_img, out_original, max_activation_maps, self.maxBack

    def meanGB(self,im,desired_class=None,iters=5):
        res = []
        for i in range(iters):
            im2 = np.array(im) / 255 + np.random.normal(0, 0.02, np.array(im).shape)
            im2 = Image.fromarray(np.clip((im2 * 255), 0, 255).astype(np.uint8))
            img_salida, img_salida_copy, max_activ, max_back = self.getGB(im2,desired_class)
            res.append(img_salida)

        img_mean = np.mean(np.array(res), axis=0)
        return img_mean


def pintaI(img,color=None, normalize=False, interpolation='nearest'):
    if normalize:
        img = img.copy()
        img -= img.min()
        img /= img.max()

    if color is not None:
        plt.imshow(img,cmap=color, interpolation=interpolation)
    else:
        plt.imshow(img)
    plt.show()

# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
#   labels: [String] List of labels
#   cols: Number of columns
def pintaMISingleMPL(vim,labels=None,title="",cols=3, ratio=0.5, interpolation = 'bilinear', color=None):
    # Transform every image to RGB
    # vim2 = np.array([
    #     cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    #     if len(im.shape) == 2
    #     else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     for im in vim])
    res = []
    for im in vim:
        im -= im.min()
        im /= im.max()

        if len(im.shape) == 2 and color is None:
            im = np.stack([im, im, im], axis=2)
        res.append(im)
    vim = np.array(res)

    # Calculate ncols and nrows
    import math
    cols = min(cols,len(vim))
    rows = math.ceil(len(vim)/cols)

    fig, ax = plt.subplots(figsize=(10, int(10*ratio)))

    plt.suptitle(title)
    # Add subplot for image in vim
    for i,im in enumerate(vim):
        plt.subplot(rows,cols,i+1)
        plt.imshow(im,interpolation=interpolation, cmap=color)

        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        # Add label if needed
        if labels is not None:
            plt.title(labels[i])
    plt.tight_layout()
    plt.show()

import matplotlib.cm as cm
def mergeImages(img, hmap):
    img = img.copy()
    img -= img.min()
    img /= img.max()
    img = (img * 255).astype(np.uint8)
    hmap = hmap.copy()
    hmap -= hmap.min()
    hmap /= hmap.max()
    hmap = (cm.gnuplot2(cv2.resize(hmap, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_CUBIC))[:, :,
            :3] * 255).astype(np.uint8)
    a = 0.3
    return np.clip(img * a + hmap * (1 - a), 0, 255).astype(np.uint8)

def main():
    # Open image
    im = Image.open("imagenes/cat_dog3.jpg")
    im = ImageOps.fit(im,(224,224),Image.ANTIALIAS,0,(0.5,0.5))


    md = models.resnet34(pretrained=True)
    gBP = GuidedBackprop(md)

    # # #
    # Guided Backprop
    # # #

    #out_img, out_img_ori, max_activ, max_back = gBP.getGB(im,281)
    out_img, out_img_ori, max_activ, max_back = gBP.getGB(im,281)
    greatest_max_back = list(map(lambda x: x[0], max_back))
    # pintaI(out_img)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    img_salida_gray = rgb2gray(out_img)
    # pintaI(img_salida_gray, color=plt.get_cmap('gray'))

    # pintaI(out_img_ori)
    # pintaMISingleMPL(
    #     [out_img,img_salida_gray,out_img_ori],
    #     labels=["GuidedBP", "GuidedBP B&W", "Original"])


    # # #
    # Max activation layers
    # # #

    # Forward max activation
    # pintaMISingleMPL(max_activ,ratio=1,cols=3,interpolation='nearest')

    # Backward oriented max activation
    # pintaMISingleMPL(reversed(greatest_max_back),ratio=1,cols=3,interpolation='nearest')

    # Paint max activation per layer
    # for im_lst in max_back:
    #     pintaMISingleMPL(im_lst[:16],ratio=1,cols=4,interpolation='nearest')

    # Heatmap
    hmap_list = [np.sum(np.array(list(map(lambda x: x[1] * x[0], el))), axis=0) for el in max_back]


    pintaMISingleMPL(hmap_list, ratio=1, cols=3, interpolation='spline16', color='gnuplot2')
    pintaI(hmap_list[0],color='gnuplot2',normalize=True,interpolation='spline16')
    pintaI(mergeImages(out_img_ori,hmap_list[0]))

    # # #
    # Mean Guided Backprop
    # # #
    # 814 speedboat
    # img_mean = gBP.meanGB(im,168) #Tigercat
    # pintaI(img_mean,normalize=True)


    # # #
    # Retriever noise confussion
    # # #
    #
    # # golden retriever 207
    # img_mean1 = gBP.meanGB(im,207) #Redbone
    #
    # # flat-coated retriever 205
    # img_mean2 = gBP.meanGB(im, 205)  # Redbone
    #
    # # Chesapeake Bay retriever 209
    # img_mean3 = gBP.meanGB(im, 209)  # Redbone
    #
    # pintaMISingleMPL(
    #     [img_mean1,img_mean2,img_mean3],
    #     labels=['Golden Retriever', 'Flat-Coated retriever', 'Chesapeake Bay Retriever'])
    a=0



if __name__ == "__main__":
    main()