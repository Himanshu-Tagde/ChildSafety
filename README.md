# ChildSafety
 The dataset consists of images labeled as either 'good' or 'bad' environments, stored in separate folders. The TensorFlow ImageDataGenerator class is employed to load and preprocess these images, resizing them to 150x150 pixels, normalizing pixel values, and applying augmentation techniques like rotation and shifting to help the model generalize better.
CNNs are ideal for image classification because they automatically learn the spatial hierarchies of features such as edges, textures, and shapes. The model starts with three convolutional layers to extract features, followed by pooling layers to reduce the spatial dimensions. After flattening the feature maps into a 1D vector, the model passes this information to fully connected layers that perform the classification. The final output layer uses a sigmoid activation function to classify the environment as either 'Good' or 'Bad.'
Once the model architecture is defined, the training process begins. The model is compiled using binary crossentropy as the loss function and the Adam optimizer for efficient training. The model is then trained on the dataset for several epochs, with the training accuracy being monitored to ensure the model is learning effectively.
During the initial training of the Child Environment Quality Predictor model, I ran into a significant issue with overfitting. While the model performed well on the training data, it struggled to generalize to the validation set, meaning it was essentially memorizing the training data rather than learning patterns that would apply to new, unseen images.
So, I decided to introduce data augmentation. By applying techniques like random rotations, shifts, and flips to the training images, I effectively increased the size and variety of the dataset. This helps the model learn to recognize patterns and features under different conditions, rather than just memorizing the specific details of the images in the training set.
