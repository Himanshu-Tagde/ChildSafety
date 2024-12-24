# ChildSafety
Purpose: Predicts whether the environment surrounding a child is "Good" or "Bad" based on images of the environment.

Dataset:
Images are organized into two folders: good and bad environments.
Separate folders for training and testing datasets:
Training set: /content/new_train/
Test set: /content/new_test/

Preprocessing:
Images are resized to 256x256 pixels.
Pixel values are normalized to the range [0, 1] for consistency in training.

Model Architecture:
Convolutional Neural Network (CNN) used for image classification.
Three convolutional layers with ReLU activation to extract features like edges and textures.
Batch Normalization applied to stabilize training and reduce overfitting.
MaxPooling after each convolutional layer to reduce spatial dimensions and focus on important features.
Flattening the feature maps to convert the 2D data to 1D for fully connected layers.
Two fully connected layers with ReLU activation.
Dropout added after each dense layer (with a rate of 0.1) to prevent overfitting by randomly dropping neurons during training.
Final dense layer with sigmoid activation for binary classification (outputting a value between 0 and 1).

Loss Function and Optimization:
Binary Crossentropy loss function for binary classification (good vs. bad).
Adam optimizer for efficient training.


Training Process:
Model is compiled with the Adam optimizer and binary crossentropy loss.
Training uses a batch size of 32 and images are resized to 256x256.
Data augmentation techniques like random rotations, shifts, and flips are applied to the training set to increase dataset size and improve robustness.

Overfitting and Improvements:
Batch Normalization and Dropout used to address overfitting during initial training.
Despite these techniques, performance on the validation set remained stagnant, so data augmentation was added to improve generalization and model robustness.


Scoring System:
For each prediction:
+5 points for a correct prediction (model's prediction matches true label).
-5 points for an incorrect prediction (model's prediction does not match true label).
The score is updated after every prediction to encourage continuous learning.


Summary:

The dataset consists of images labeled as either 'good' or 'bad' environments, stored in separate folders. The TensorFlow ImageDataGenerator class is employed to load and preprocess these images, resizing them to 150x150 pixels, normalizing pixel values, and applying augmentation techniques like rotation and shifting to help the model generalize better.
CNNs are ideal for image classification because they automatically learn the spatial hierarchies of features such as edges, textures, and shapes. The model starts with three convolutional layers to extract features, followed by pooling layers to reduce the spatial dimensions. After flattening the feature maps into a 1D vector, the model passes this information to fully connected layers that perform the classification. The final output layer uses a sigmoid activation function to classify the environment as either 'Good' or 'Bad.'
Once the model architecture is defined, the training process begins. The model is compiled using binary crossentropy as the loss function and the Adam optimizer for efficient training. The model is then trained on the dataset for several epochs, with the training accuracy being monitored to ensure the model is learning effectively.
During the initial training of the Child Environment Quality Predictor model, I ran into a significant issue with overfitting. While the model performed well on the training data, it struggled to generalize to the validation set, meaning it was essentially memorizing the training data rather than learning patterns that would apply to new, unseen images.
So, I decided to introduce data augmentation. By applying techniques like random rotations, shifts, and flips to the training images, I effectively increased the size and variety of the dataset. This helps the model learn to recognize patterns and features under different conditions, rather than just memorizing the specific details of the images in the training set.
