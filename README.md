# Paddy-Disease-Classification-Using-Deep-Learning

Paddy, an unhusked form of rice, is prone to several diseases sometimes, leading to a loss of 70% of yield. Manual supervision of paddy diseases is a costly and time-consuming process. So a reliable method that saves farmers time and money is highly demanded. With the advancement of deep learning and computer vision, we implemented a multiclass classification technique to classify the given paddy leaf into ten classes out of 9 disease classes and one regular class, the nine disease classes, are bacterial leaf blight, bacterial leaf streak, bacterial pinnacle blight, blast, brown spot, dead Heart, downy Mildew, hispa, and tungro. We developed four distinct deep learning models to address this issue, one of which, Vision Transformers, performed excellently and attained a validation accuracy of 91.30%.

# INTRODUCTION

Rice, or Oryza sativa, is a husked kind of paddy grown in flooded fields in Asia's southern and eastern parts. In 2021-2022, China consumed the most rice (154.9 million metrics), followed by India (103.5 million metrics). A decrease in paddy output might result in food scarcity and disrupt the supply chain. Faster diagnosis of paddy diseases can help farmers take the necessary precautions to prevent crop loss. We worked on the paddy disease classification problem for this project and created a multiclass deep transfer learning model. This research aims to determine if the paddy leaf images provided are normal or diseased. So, we created four different models using deep learning and computer vision. Validation accuracy for CNN (base model), mobilenetv2, inceptionv3, and Vision Transformers was 87.5 percent, 83.10 percent, 93.1 percent, 93.7 percent, 94.1 percent, and 91.30 percent, respectively. 


## A. Comprehension Of Each Paddy Disease In The Dataset

#### Bacterial Leaf Blight: 

The disease was caused by the Gram-negative bacterium Xanthomonas oryzae PV. When the plant is affected by the disease, the leaves will turnyellowish-white, and the leaves will become dry from the back tip to curling.


#### Bacterial Leaf Streak: 

The disease is caused by Xanthomonas oryzae PV. The disease will cause the browning and drying of leaves in the paddy plants.


#### Bacterial Pinnacle Blight: 

The disease was caused by Burkholderia glumae. The disease will cause the rotting of the rice grains.

#### Blast: 

The disease was caused by a fungus auricularia oryzae. The disease affects the leaves, collars, necks, pinnacles, and seeds of the paddy plant. The paddy plant's leaves affected by this disease will observe elliptical-shaped spots with gray centers and red-brown margins.

#### Brown Spot: 

The disease was caused by fungus cochliobolus miyabeanus; the disease will affect the coleoptile leaves, leaf sheath, panicle branches, glumes, and spikelets, paddy plants affected with the disease will have an oval-shaped spot with brown margin.

#### Dead Heart: 

The disease was caused by a stem borer. The disease can be observed when the youngest partially leaves of the plants become white and die.

### Downy Mildew: 

The disease was caused by the fungus Peronospora parasitica. When the paddy crop is affected by the disease, angular spots of yellow color can be observed on the upper surface of the leaf.



#### Hispa: 

The disease was caused by an insect called rice hispa. It is a  bluish-black beetle the crop affected with this disease will have irregular translucent white patches parallel to the leaf veins.


#### Tungro: 

The disease was caused by two different viruses transmitted by leafhoppers; the affected paddy crop can be observed with orange-yellow color.


# DATA

The data we are using for this project is collected from an ongoing Kaggle competition named paddy disease classification. The data set has ten classes discussed in the report's introduction. The dataset consists of 10,406 images. 


Images have the following characteristics: The image has a height of 640 pixels and a width of 480 pixels, with a maximum pixel value of 255.0, the minimum pixel is 0.0, the mean pixel value is 115.9670, and the standard deviation is 71.6155.



#### IMAGE RESIZING
To reduce the computational costs and get all images of the dataset into one size for modeling, we resize all of the images into 256x256.

#### DATA NORMALIZATION
The input image pixels are in the [0, 255] range to be normalized into the [0, 1] range. To make the model learn faster makes the model convergence faster.

# DATA AUGMENTATION

We can see from the figure above that some classes have a lot of data imbalance. To solve this problem, we placed a data augmentation layer on top of each model to avoid overfitting. Operation performed in Augmentation Layer:

Rotation_range = 5
Shear_range = 0.3
Zoom_range = 0.3
Width_shift_range = 0.05
Height_shift_range = 0.05
Horizontal_flip = True
Vertical_flip = True


# TRAIN-TEST SPLIT

The dataset has been divided into training and testing using image data generators from Keras library of sizes 70:30 ratio, which creates mini-batches of a batch size of 64. This concludes that 7288 Images are in the Training dataset. 3119 Images are in the testing dataset.

# METHODS
To solve this problem, we initially created a base model without any pre-trained architecture, followed by a four-deep transfer learning model and a transformer model used to attain higher accuracy.

#### CNN

Convolutional Neural Networks are special kinds of multi-layer neural networks that are used to find the visual patterns in the images with minimal data pre-processing. This project's base model consists of a couple of conv2d, maxpooling2d, and dense layers with a relu activation function and a 3x3 kernel size.


#### VGG16

VGG16(visual Geometry group) is a convolutional neural network model with 16 different layers containing tune-able parameters except for the max-pooling layer. It played a vital role in the improvement of computer vision. Karen Simonyan and Andrew Zisserman proposed VGG16 at the University of Oxford in 2013, but the actual model was submitted during the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) competition in 2014 first runner up in that competition. 

#### MOBILENETV2

Mobilenetv2 is a pre-trained deep transfer learning model created by Google and designed for mobile and resource-constrained environments. Mobilenetv2 is built upon the Mobilenetv1 architecture, which uses light depth-wise separable convolution as efficient building blocks to extract features followed by inverted residual structure (skip or residual connections between bottleneck layers). As a result, the mobilenetv2 architecture consists of 32 conv2d layers and 19 skip or residual bottleneck layers.

#### INCEPTIONV3

InceptionV3 is a convolutional neural network trained on more than a million images from the imagenet dataset. It is the 3rd edition of the CNN model by google. Instead of using a single filter size in a single image block, it uses multiple types of filter sizes, which are then concatenated and passed to the next layer.

#### DENSENET121

DenseNet121 is a CNN architecture that focuses on making the deep learning networks go even deeper while making them more effective to train by using shorter connections between the layers. DenseNet121 incorporates a couple of operations like Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifier to propagate label information lower down the network.

#### VISION TRANSFORMER

Vision transformers solve computer vision problems that work similarly to NLP transformers. The vision transformers use patches of images as token and attention mechanisms to find the relationship between Input token pairs. This can be done with the help of CNN or by replacing some components of CNN. With the help of vision transformers, we can solve image classification. The following figure shows the procedure of the vision transformer for classification.


# EXPERIMENTS

Each model in this project is trained for 20 epochs. The loss function was used as a categorical cross-entropy because it was a multiclass classification problem, and the optimizer was Adam with a learning rate of 3e-4, which was decayed based on the validation accuracy with the help of callbacks. A total of three checkpoints or callbacks were created. The first one was TerminateNaN: The callbacks stopped the training if the loss was NaN. ModelCheckPoint: It saves the model after every epoch if the validation accuracy improves from the previous epoch. ReduceLRonPlateau: If our validation accuracy at that epoch is less than the previous epoch accuracy, we decrease the learning rate by 10% Categorical_accuracy was used as a metric to evaluate the models. The categorical accuracy is similar to the accuracy, whereas categorical calculates the percentage of similarities between the actual and predicted one-hot encoded labels.

#### RESULT OF BASELINE CNN MODEL

Training Accuracy
93.76%
Validation Accuracy
87.5%

#### RESULT OF MOBILENETV2 MODEL

Training Accuracy
97.12%
Validation Accuracy
93.7%

#### RESULT OF INCEPTIONV3 MODEL

Training Accuracy
98.09%
Validation Accuracy
94.1%

#### RESULT OF  VISION TRANSFORMERS MODEL

Training Accuracy
93.82%
Validation Accuracy
91.30%


All the pre-trained transfer learning models are initially not trained fully, which means we made layers frozen and used the imagenet weights, but that does not give us good results. The models are not able to converge. So we trained all parameters on top of imagenet weights which results are shown above which makes us satisfactory.

# CONCLUSION

In this study, we proposed a couple of variations of transfer learning models like vgg16, densenet121, inceptionv3, mobilenetv3, and a transformer model called vision transformer to perform paddy disease classification. We compared five models of different combinations. Therefore, vision transformers yield more minor losses and avoid overfitting the model, which returns an accuracy of 91.30%. As a Future work, we create GANs to solve the problem of imbalance in data which helps us increase model performance, and we deploy the model in the google cloud to create a small react native app that helps farmers detect paddy diseases with an expert intervention.


































