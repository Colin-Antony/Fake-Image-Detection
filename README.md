# Fake-Image-Detection
Fake Image Classifier that distinguishes real images from fake images.  

## Dataset  
For this problem i used the CIFAKE dataset. Which is a dataset consisting of 120,000 images.  
The CIFACE dataset has 60,000 real and 60,000 fake images. The training set has 100,000 images and the test set has 20,000 images.  
Here is what the dataset description says on kaggle about where they obtained the images from:  
"For REAL, we collected the images from Krizhevsky & Hinton's CIFAR-10 dataset"  
"For the FAKE images, we generated the equivalent of CIFAR-10 with Stable Diffusion version 1.4"  
The fake images were generated by Stable diffusion which is a type of Generator Model.  
Thus dataset has images of size 32x32. There is a paper that talks about how CNNS can learn features even from these small images that we(humans) can barely understand. Here is that paper:  
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.  
[Link to the dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)  
 

## Model, Layers, Optimizer, Loss functions  
1) Choose a Sequential Model.  
2) Chose a CNN architecture as we are dealing with images. Used Convolutional layers along with Pooling Layers(MaxPool2D).  
3) Passed on the features extracted by the Convolutional layers onto a simple Neural Network after flattening it to a 1-D vector.  
4) Loss: Binary Crossentropy as it is very common for binary classification problems. It measures the dissimilarity between the predicted probabilities and the actual binary labels (0 or 1).  
5) Optimizer: RMSprop was used. Adam would work well too. No particular reason for choosing one over the other.  


## Performance
Model was run on the GPU for only 5 epochs. Here are the performance numbers after 5 epochs. Increasing the epochs wouldnt matter and would only lead to overfitting in this case.  
1) Loss - Training: 0.2650 Validation: 0.2249  
2) Accuracy - Training: 0.9017 Validation: 0.9143  
3) Precision - Training: 0.9025 Validation: 0.9125  
4) Recall - Training: 0.9008 Validation: 0.9165  
These numbers are present in the output document.  


### Output  
The output can be viewed in full in the output document. However, here are a few screenshots too.  
