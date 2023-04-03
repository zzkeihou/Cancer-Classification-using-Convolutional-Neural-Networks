# Colon-cancer-classification
Colon Cancer Classification using CNN

Table of Contents
1. Overview
2. Data
3. Environment and tools
4. EDA
5. Model Architecture
6. Model Performance
7. Acknowledgements

## Overview
Colon cancer, also known as colorectal cancer, is a type of cancer that originates in the colon or rectum. It is one of the most common types of cancer worldwide, and early detection plays a crucial role in improving treatment outcomes. KRAS (Kirsten Rat Sarcoma viral oncogene homolog) is a gene that plays a significant role in cell signaling pathways, controlling cell growth and proliferation. Mutations in the KRAS gene are associated with various types of cancer, including colon cancer. In colon cancer patients, approximately 40% have KRAS mutations, which can impact treatment strategies and prognosis.

This project focuses on the classification of colon cancer images based on the presence or absence of KRAS mutations and the type of TCGA cancer. The images are classified into three categories: 'All_tcga' (representing a general set of TCGA cancer images), 'Colon_kras' (colon cancer with KRAS mutations), and 'Colon_non_kras' (colon cancer without KRAS mutations). This classification task aims to develop a machine learning model that can effectively differentiate between the three categories, potentially aiding in better diagnosis and treatment planning for patients.

To achieve this, a convolutional neural network (CNN) model is implemented to analyze and learn features from cancer images. The entire process includes importing libraries, data preparation, exploratory data analysis (EDA), model architecture design, and model performance evaluation. By accurately classifying colon cancer images based on KRAS mutations and cancer types, this project contributes to the broader goal of improving colon cancer diagnosis and treatment.

## Data
The data can be download from https://www.cancerimagingarchive.net/collections/

Label 1: 280 colon kras data (7 cases)

Label 2: 400 colon non kras data (10 cases)

Label 3: Randomly selected 1000 data from all TCGA cancer, which include:

TCGA-BLCA

TCGA-BRCA

TCGA-CESC

TCGA-ESCA

TCGA-KIRC

TCGA-KIRP

TCGA-LIHC

TCGA-LUAD

TCGA-LUSC

TCGA-OV

TCGA-PRAD

TCGA-READ

TCGA-SARC

TCGA-STAD

TCGA-THCA

TCGA-UCEC

## Environment and tools
- Colab notebook
- NumPy
- Pandas
- Matplotlib 
- Seaborn
- scikit-image
- Keras

## EDA
Pie charts showing the distribution of the images in each category and visualization of random samples of non-kras and kras images
![image](https://user-images.githubusercontent.com/116041838/229614486-a8067528-1339-4df3-a2d4-30d7b27bc580.png)

## Model Architecture
![image](https://user-images.githubusercontent.com/116041838/229646453-06ce4874-79b6-4c1d-8a62-29bcd067f770.png)
This figure illustrates the architecture of the Convolutional Neural Network (CNN) model for colon cancer classification. The grayscale images with a size of 256x256 pixels are fed into the model, which then processes them through multiple interconnected layers. The model consists of an input layer, followed by alternating Conv2D and MaxPooling2D layers, with BatchNormalization applied after each convolution operation. The Conv2D layers use ReLU activation functions and progressively increase the number of filters (32, 64, 128) to capture intricate patterns in the data. The MaxPooling2D layers have a pool size of 2x2 and help reduce the spatial dimensions while preserving important features.

Dropout layers are introduced after certain MaxPooling2D layers to prevent overfitting by randomly setting some activations to zero during training. The feature maps are then flattened to a 1D array, which is processed by a Dense layer with 128 units and ReLU activation, followed by another BatchNormalization and Dropout layer. Finally, the output Dense layer consists of a single unit with a sigmoid activation function to perform binary classification between colon kras and colon non-kras cancer types. The model is trained and optimized using the Adam optimizer and binary cross-entropy loss function, with accuracy as the primary evaluation metric.

## Model Performance

## Acknowledgements
