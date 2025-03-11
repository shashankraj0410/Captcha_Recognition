# Captcha_Recognition
This project focuses on captcha text recognition using CRNN (Convolutional Recurrent Neural Network). It leverages deep learning techniques to accurately decode text from CAPTCHA images, which are commonly used for security verification.

#Dataset

We use the CAPTCHA Version 2 Images dataset from Kaggle.

Dataset Link: https://www.kaggle.com/datasets/fournierp/captcha-version-2-images

The dataset contains a variety of CAPTCHA images with corresponding text labels.


#Tech Stack

*Python
*PyTorch (For deep learning model)
*Torchvision (For handling image datasets)
*OpenCV (For image preprocessing)
*NumPy & Pandas (For data manipulation)
*Matplotlib & Seaborn (For visualization)

#Features

*Preprocessing of CAPTCHA images (grayscale conversion, thresholding, resizing, augmentation).
*Training a CRNN model for sequence-to-sequence text recognition using PyTorch.
*Use of CTC Loss (Connectionist Temporal Classification) for sequence decoding.
*Model evaluation with accuracy metrics and visualizations.
*Inference on new CAPTCHA images.



#Model Architecture

*The CRNN (Convolutional Recurrent Neural Network) model consists of:
*CNN Layers (ResNet-18 backbone) for feature extraction from images.
*RNN Layers (Bidirectional LSTM) for sequential text processing.
*CTC Loss Function for alignment-free text recognition.
