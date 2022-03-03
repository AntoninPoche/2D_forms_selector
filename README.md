# 2D Forms Selector

This repository allow the creation of a dataset and a model adapted to this dataset. 



## The dataset

The dataset is a classification dataset of images, the images have four parts and each part have a shape. One of the parts is highlighted (selected), the selected shape is the one we want to predict.
![images_examples](https://user-images.githubusercontent.com/90199266/156572324-2f774abd-2490-4f71-b496-74f9ec932074.png)



## The model

The model was built specifically for this dataset and every weights were set manually. The objective is to have a neural network that we understand to evaluate explainability methods.
We will compare the explanations with the functionment we know and take a step back on what explanation methods really provide.

The model is built with four blocks:
- the first one that separates the different part of the image,
- the second that find which is the part to look at,
- the third in parallel of the second that find shapes scores for each part,
- the fourth one using both previous part to selected the shapes scores.
![model_graph](https://user-images.githubusercontent.com/90199266/156572349-a45b97f4-bb80-4a72-836a-4f563a6ff52e.png)
