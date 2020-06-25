# Brain-Tumor-Detector
In this project, a Convolution Neural Network (CNN) has been used to detect a tumour through brain Magnetic Resonance Imaging (MRI) images
in Python. Data Augmentation applied to take care of class imbalance nature of the data and then Images were first applied to the CNN to
classify whether an MRI Image belongs to a Benign or Malignant Category.<br>

Libraries Used :Tensorflow, Keras,cv2,sklearn,Imutils,Numpy,Matplotlib,OS.<br>

 [Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).<br>


## Data Augmentation:

Data has been upsampled via Augmentation to take care of class imbalance.

| <!-- -->  | Positive       | Negative |
| --------- | -------------- | -------- |
| Before    | 155            | 98       |
| After     | 1085           | 980      |


## Data Preprocessing

For every image, part of the tumor cropped , resized and applied to a gray scale 0-1.<br>

## Data Split:

The data was split in the following way:
1. 70% of the data for training.
2. 15% of the data for validation.
3. 15% of the data for testing.

# Neural Network Architecture and Training


Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the following layers:<br>

1. A Zero Padding layer .
2. A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1.
3. A batch normalization layer .
4. A ReLU activation layer.
5. Two consequtive Max Pooling layers.
6. A flatten layer  to flatten it into a 1D Vector.
7. A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).

for the above architecture model is trained for 20 epochs.


# Results

Now, the best model (the one with the best validation accuracy) detects brain tumor with:<br>

**88.7%** accuracy on the **test set**.<br>
**0.88** f1 score on the **test set**.<br>
These resutls are very good considering that the data is balanced.

**Performance table of the best model:**

| <!-- -->  | Validation set | Test set |
| --------- | -------------- | -------- |
| Accuracy  | 91%            | 89%      |
| F1 score  | 0.91           | 0.88     |


<br>Thank you!



