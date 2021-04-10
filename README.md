# Brain Tumor Detection

<img src="/images/header.jpg" width="550">

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
Although the data comes seperated into subdirectories than be directly loaded into an ImageDataGenerator object, they were read in using [OpenCV](https://docs.opencv.org/master/) and put into TensorFlow Datasets for practice. The 2870 training images are divided into a training and validation set using a 20% split, while the 394 testing images remain unsplit. We have also augmented the training images to apply random rotations, brightness, and contrast to strength the models training. But the valiation and testing images are only converted to tensors and normalized. Below are the first 25 examples from the training split after augmentation:

<img src="/images/train_images.jpg" width="770">

The training and validation images are converted into *tfrecords* files in order to be uploaded to **S3**, while the testing images remain as a TensorFlow Dataset since they will be used within the notebook on the deployed model (not uploaded to S3).

--- 

## Modeling

---

*README will be updated as project is worked on...*
