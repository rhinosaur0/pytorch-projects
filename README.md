# PyTorch practice

Small PyTorch projects I worked in

Cognate classifier: identifies cognates in Spanish and English
* Uses LMST neural network on character embeddings for words
* Uses standard sigmoid activation with BCELoss
* Trains from a dataset I organized which includes 73 non-cognate pairs and 87 cognate pairs


AI Image Detection: identifies whether an image is AI generated or not
* Uses a pre-labeled kaggle dataset to train a model that identifies AI images
* Uses 4 layers of conv2d layers (TinyVGG)
* Data augmentation causes the final model to have an accuracy of 87.5%
