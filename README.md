# Brain Tumor Detection

<img src="/notebook_images/header.jpg" width="550">

---

## Project Introduction & Goals
MRI images are one of the main tools used for analyzing tumors in the human brain. Huge amounts of image data are generated through these scans, which need to be examined by a radiologist, and can be susceptible to diagnosis error due to complex MRI images. This is where the application of neural networks come in. Through CNN's, we are able to process images in order to extract features (such as tumors) from images that can help us correctly classifying the data. The purpose of this project will be to deploy a deep learning model using Amazon SageMaker that can accurately classify MRI brain tumor images into four different categories:
- Glioma - a tumor made of astrocytes that occurs in the brain and spinal cord
- Meningioma - a usually noncancerous tumor that arises from membranes surrounding the brain & spinal cord.
- None - no tumor present in brain.
- Pituitary - a tumor that forms in the pituitary gland near the brain that can change hormone levels.

### Data
The dataset comes from [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri), but in order to move images directly into Amazon SageMaker they were clone from the original [GitHub](https://github.com/sartajbhuvaji/brain-tumor-classification-dataset) repository.

#### Data Formatting
Although the data comes seperated into subdirectories that could be directly uploaded to S3 and read into an ImageDataGenerator object, they were read in using [OpenCV](https://docs.opencv.org/master/) and put into a TensorFlow Datasets to be converted to TFRecord files. The 2870 training images are divided into a training and validation set using a 20% split (2296 and 574, repsectively), while the 394 testing images remain unsplit. We have also augmented the training images to apply random rotations to strength the models training. The images were not normalized since we used EfficientNet for our model and batch normalization occurs within the layers (initially the images were normalized and this resulted in a 90% train and 10% validation accuracy). Below are the first 25 examples from the training split after augmentation:

<img src="/notebook_images/train_images.jpg" width="770">

--- 

## Modeling
#### Setup
Both the training and model files are within the **scripts** directory. The *model* script creates our [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function), replacing the output with a dropout and new dense layer to handle our 4 classes. Below is the function used to create the model using the trained weights from ImageNet.

<img src="/notebook_images/model_summary.jpg" width="700">

The *train* script contains the Python file used to load the data from S3, convert from a TFRecord file back into a Dataset, and train/validate the model. The epochs and batch size are passed as hyperparameters from the SageMaker TensorFlow object within the main notebook. Both Early Stopping and Learning Rate Reduction are implemented for the model, with learning rate being reduced 3 times and early stopping occuring at epoch 16 when validation loss plateaued. Class weights were also computed since there is a slight difference in the *no_tumor* class compared to the other three and we need the model to learn equally from each class. They are as follows:

| Class      | Weight     | 
| ---------- | ---------- | 
| Glioma     | 0.86445783 |
| Meningioma | 0.85928144 |
| No Tumor   | 1.81072555 |
| Pituitary  | 0.88717156 |

#### Training Results
On epoch 1, the initial training accuracy was 85.1% with a validation accuracy of 62.37%. After epoch 5, 13, and 15 the learning rate was reduced from an initial value of 0.001 to a final value of 0.000008. Early stopping ended our model training after epoch 16, where the validation loss plateaued aroung 0.044. The final results from training are:

| Dataset    | Loss   | Accuracy |
| ---------- | -----  | -------- |
| Training   | 0.0036 | 99.91%   |
| Validation | 0.0431 | 98.61%   |

With such a high training accuracy, I would be skeptical that the model is overfitting. But since are validation accuracy is within 1.5% of the training accuracy, it leads me to think that the model is performing well. This will be either confirmed or denied in the testing results section depending on the accuracy of the model of predicting with new data. The final model is saved to the default S3 bucket, which will be loaded back into the main notebook and used for predicting in the next section.

#### Testing Results
Testing images/labels (394 total) were loaded and saved into numpy arrays, which were then flattened and sent to the endpoint for predicting. The maximum probability from the predicted array was then taken and converted back into a string label corresponding to the true label. With a training/validation accuracy, I expected the testing accuracy to be a similar value. However, the final result was:

| Dataset   | Accuracy |
| --------- | -------- |
| Testisng  | 73.86%   |

Seeing how the accuracy for the testing set is much worse than the training/validation sets, we needed to explore further to see which classes were having issues. The results are as follows:

<img src="/notebook_images/test_results.jpg" width="420"> <img src="/notebook_images/per_class_acc.jpg" width="420">

Looking at the confusion matrix and per class accuracy above, we can see that the *glioma* and *pituitary* classes seem to be having issues with being seperated from the other classes. We seem to be getting around 100% accuracy for both meningioma and no tumor classes. However, pituitary hs around 65% accuracy and glioma has around 25% accuracy. On top of this, most the the misclassified images seem to have high prediction probabilities associated with them (90%+, refer to end of notebook for graph). However, other notebooks also seem to be getting a test accuracy ranging from 40% to 80%, so our model seems to be doing well in comparison to those. This leads me to believe that there could be an issue with the test set that is casuing this, since using the deployed model to predict on the validation set resulted in 99% accuracy (again, this could just be overfitting if our validation set is too similar to the training set).

---

