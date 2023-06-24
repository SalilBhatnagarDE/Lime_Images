# -*- coding: utf-8 -*-
""""""
"""Interpretability

We will work on the model interpretability. 
First, we will implement LIME (Local interpretable model-agnostic explanations).

It is necessary to install a package manager e.g. [conda](https://docs.conda.io/en/latest/), and [PyTorch](https://pytorch.org) framework.

The aim is to implement LIME using the information from the publication. We rely on the Inception V3 neural network. 
In addition, the focus would be on analysing the top 1 and top 2 predictions. The focus are images.

"""

#loading libraries
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

#Read images and resize them
def load_imgs(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299,299))
    return img

#Load inception_v3 model from torchvision.models
def make_inception_v3():
    inception_v3 = torchvision.models.inception_v3(pretrained=True)

import urllib
import json

# Download the class names for ImageNet
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = urllib.request.urlopen(url)
class_index = json.loads(response.read())

# Get the class names for Inception V3 model
class_names = [class_index[str(k)][1] for k in range(len(class_index))]

# Make custom dataset and Dataloaders
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

# Make function to get top2 predictions output from inception v3
def get_top2_predictions(imgs_in_batches):
    dataset = CustomDataset(imgs_in_batches)
    dataloader_imgs = DataLoader(dataset, batch_size=10)
    predictions = torch.empty((0, 1000))
    inception_v3.eval()
    for batch in dataloader_imgs:
        output = inception_v3(batch)
        predictions = torch.cat((predictions, output), 0)

    predictions = torch.nn.functional.softmax(predictions, dim=1)

    top2_indices = torch.topk(predictions, k=2, dim=1)[1]
    top2_probabilities = torch.topk(predictions, k=2, dim=1)[0].detach()

    top1_probability = top2_probabilities.squeeze()[:,:1]
    top2_probability = top2_probabilities.squeeze()[:, 1:]

    top2_prediction = top2_indices.squeeze()[:, 1:]
    top1_prediction = top2_indices.squeeze()[:, :1]

    return top1_prediction, top2_prediction, top1_probability, top2_probability

# Make function to get names of top 2 predictions
def get_top2_names_of_predictions(top1_prediction, top2_prediction):
    for i in range(top1_prediction.shape[0]):
        print("\nThe top1 label for image",i," is:", class_names[top1_prediction[i].numpy()[0]])
        print("The top2 label for image",i," is:", class_names[top2_prediction[i].numpy()[0]])

#Import libraries for segmentation

import skimage
from skimage.segmentation import quickshift

# using skimage.segmentation, we can use quickshift to make different segments of the image

def segment(imgs):
    segments_of_images = []
    superpixels_values = []
    for i in range(imgs.shape[0]):
        segment_img = quickshift(imgs[i], kernel_size=10, max_dist=15, ratio=0.7)
        segments_of_images.append(segment_img)
        superpixels_values.append(np.unique(segment_img))
    return np.asarray(segments_of_images), (superpixels_values)

# Generate superpixels for each image
def get_superpixels(segments_of_images, superpixels_values):
    superpixels = [None]* int(segments_of_images.shape[0])
    for j in range(segments_of_images.shape[0]):
        superpixels[j] = []
        for i in range((superpixels_values[j].shape[0])):
            mask = (segments_of_images[j] == superpixels_values[j][i])
            superpixels[j].append((np.multiply(original_imgs[j], np.stack([mask] * 3, axis=-1))))
    return superpixels


# Geting perturbed images for each image from superpixels made for each image
def make_pertubed_imgs(original_imgs, number_of_perturbed_imgs=300, superpixels_values=None):
    perturbed_imgs = [None] * (original_imgs.shape[0])
    index_perturbed_imgs = [None] * (original_imgs.shape[0])

    number_of_perturbed_imgs = number_of_perturbed_imgs

    for m in range(original_imgs.shape[0]):
        perturbed_imgs[m] = [None] * (number_of_perturbed_imgs)
        index_perturbed_imgs[m] = np.zeros((number_of_perturbed_imgs, superpixels_values[m].shape[0]), dtype=np.uint8)
        for j in range(number_of_perturbed_imgs):
            perturbed_imgs[m][j] = np.zeros(original_imgs[m].shape, dtype=np.uint8)
            for i in range((superpixels_values[m].shape[0])):
                a = int(np.round(np.random.rand(),0))
                b = ((superpixels[m][i] * a ))
                if a==1:
                     index_perturbed_imgs[m][j][i] = 1
                perturbed_imgs[m][j] = perturbed_imgs[m][j] + b

        perturbed_imgs[m] = np.asarray(perturbed_imgs[m])

    return perturbed_imgs


# Make function to get predictions output from inception v3 for Perturbed Images

def perturbed_get_1_2_predictions(imgs_in_batches, top1_predictions_of_original_images,
                                  top2_predictions_of_original_images, image_num):


    dataset = CustomDataset(imgs_in_batches)
    batch_size = 50
    dataloader_imgs = DataLoader(dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    predictions = torch.empty((0, 1000)).to(device)
    inception_v3.eval().to(device)

    with  torch.no_grad():
        for batch in dataloader_imgs:
            batch = batch.to(device)
            output = inception_v3(batch)
            predictions = torch.cat((predictions, output), 0)

    predictions = torch.nn.functional.softmax(predictions, dim=1)

    first_predicted_probability = predictions[:,top1_predictions_of_original_images[image_num]:top1_predictions_of_original_images[image_num]+1]
    print("The top1 prediction of image 1 was:", top1_predictions_of_original_images[image_num])

    second_predicted_probability = predictions[:,top2_predictions_of_original_images[image_num]:top2_predictions_of_original_images[image_num]+1]
    print("The top2 prediction of image 1 was:", top2_predictions_of_original_images[image_num])

    return first_predicted_probability.cpu().numpy(), second_predicted_probability.cpu().numpy()


from sklearn.metrics.pairwise import cosine_similarity

def make_surrogate_data(original_image, index_perturbed_imgs, perturbed_imgs,
                        perturbed_predictions_top1, perturbed_predictions_top2):
    x_data = index_perturbed_imgs
    y_label_top1 = perturbed_predictions_top1
    y_label_top2 = perturbed_predictions_top2

    original_image = np.reshape(original_image, (1, 299, 299, 3))
    sample_weights = cosine_similarity(original_image.reshape(1, -1), perturbed_imgs.reshape(300, -1))
    sample_weights = sample_weights.reshape(-1, 1)
    sample_weights = sample_weights.ravel()

    return x_data, y_label_top1, y_label_top2, sample_weights


"""## Training Linear Regression model and getting top most relevant 3 superpixels for each Image"""

# Training lInear Regression model and getting most relevant 3 superpixels for each Image

from sklearn.linear_model import LinearRegression

def get_explainability(x_data, y_data, sample_weight, image_number):
    lin_reg = LinearRegression()
    lin_reg.fit(x_data, y_data, sample_weight=sample_weight)
    coefficients = lin_reg.coef_
    sorted_indices = np.argsort(coefficients)[0][::-1]
    top5_sorted_indices = sorted_indices[:3]

    return plt.imshow(superpixels[image_number][top5_sorted_indices[0]]
          +superpixels[image_number][top5_sorted_indices[1]]
          +superpixels[image_number][top5_sorted_indices[2]]
          )




