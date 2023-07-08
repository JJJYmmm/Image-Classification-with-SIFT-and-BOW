# Image-Classification-with-SIFT-and-BOW

A Image Classifier with SIFT,BOW and SVM

Program Structure：

- main.py : Program entry point
- DataProcess.py : Read the image and extract the SIFT features of the image
- ImageInfo.py : The class of image related information, including category, size, location of SIFT features, and SIFT descriptors
- Vocabulary.py : Bag-of-words model, clustering all SIFT features to get words, and extracting SPM features of each image
- ClassifierKernel.py : Implementation of SVM Model

See **output.txt** for the results

There is test data's confusion matrix.

[confusion matrix](https://github.com/JJJYmmm/Image-Classification-with-SIFT-and-BOW/blob/master/MyBoW/confusion_matrix.png)
