# Swin ViT Lung Nodule Classifier

This repository was built as a small demonstration of my coding skills, data management and applied knowledge of deep learning techniques. The selected task is derived from the Luna16 challenge. The task is to classify whether clinical follow-up is required, based on the provided coordinates of a specific lung nodule candidate. Due to the limited time which was available to me, the intended purpose of this project is not neccesarily to create a high-performance solution. Instead, it aims to technically implement a more creative solution. 

## Model

I used a pretrained [Swin UNETR model](https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR), which was trained to segment brain tumours in CT scans by [Hatamizadeh et al.](https://arxiv.org/abs/2201.01266). The architecture of this model is displayed in Figure 1. I used the pretrained ViT-encoder to extract features from the image and removed the rest of the model. Furthermore, I designed a MLP-head which was used to classify the clinical follow-up, based on these image features as shown in Figure 2.

__Figure 1__

![Figure 1](https://raw.githubusercontent.com/larsleijten/swin_classifier/main/imgs/swin_unetr.png "Figure 1")

__Figure 2__

![Figure 2](https://raw.githubusercontent.com/larsleijten/swin_classifier/main/imgs/my_model.png "Figure 2")


## Data

The candidates dataset consisted of about 700,000 candidates, with approximately 1500 of those requiring follow-up. To speed up training times for this project, I decided to work with image patches with centered candidates. With regard to data storage, I selected all 1500 positive candidates and about 2000 negative candidates and saved the patches. The dataset was split 80:20 over a train and validation set.

## Training

Initially, I saved the feature vectors which the Swin Encoder derived from the patches to efficiently train the MLP-head before training the entire model. This method did not produce any accurate results on the validation data. Consequently, I decided to train the Swin Encoder and MLP-head fully as a combined model. 

## Results

After training for 12 epochs over the training data, the model reached a classification accuracy of 80.5% over the validation set. 

The train loss and validation accuracy are plotted in figure 3.

__Figure 3__

![Figure 3](https://raw.githubusercontent.com/larsleijten/swin_classifier/main/imgs/results%201-12.png "Figure 3")

Although model performance was not the main purpose of this project, the current result is still somewhat underwhelming. I will try to make a few adaptations in the upcoming days to increase model performance. These adaptations include tackling the exploding gradient problem I ran into and adding some data augmentations.

## My Contribution

In this project, I designed and wrote the code for:
- The datasets which load the images patches or feature vectors and associated labels
- The scripts used to generate the image patches.
- The script used to save the feature vectors derived from these images
- The MLP-head
- The model combining the Swin Encoder and the MLP-head
- The training and validation loops

The only mostly external code was the code for the Swin Encoder. This derived from code for a [Swin UNETR model](https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR), which I reduced to only include the ViT encoder.