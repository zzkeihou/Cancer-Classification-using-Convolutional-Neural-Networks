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
The data can be downloaded from https://www.cancerimagingarchive.net/collections/

Label 1: 280 colon kras samples (7 cases)

Label 2: 400 colon non-kras samples (10 cases)

Label 3: Randomly selected 1,000 samples from all TCGA cancer types, including:

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

Before using the data, it is necessary to convert the DICOM (Digital Imaging and Communications in Medicine) files into JPG format. To run the code, open the Cancer image prediction final.ipynb file.




## Environment and tools
- Colab notebook
- NumPy
- Pandas
- Matplotlib 
- Seaborn
- scikit-image
- Keras

## EDA
The distribution of the images in each category and visualization of random samples of non-kras and kras images
![image](https://user-images.githubusercontent.com/116041838/229614486-a8067528-1339-4df3-a2d4-30d7b27bc580.png)

Random samples of non-kras and kras images
![image](https://user-images.githubusercontent.com/116041838/229649713-7d5f2577-3e48-4463-8098-db63d1299636.png)


## Model Architecture
![image](https://user-images.githubusercontent.com/116041838/229646453-06ce4874-79b6-4c1d-8a62-29bcd067f770.png)

This figure illustrates the architecture of the Convolutional Neural Network (CNN) model for colon cancer classification. The grayscale images with a size of 256x256 pixels are fed into the model, which then processes them through multiple interconnected layers. The model consists of an input layer, followed by alternating Conv2D and MaxPooling2D layers, with BatchNormalization applied after each convolution operation. The Conv2D layers use ReLU activation functions and progressively increase the number of filters (32, 64, 128) to capture intricate patterns in the data. The MaxPooling2D layers have a pool size of 2x2 and help reduce the spatial dimensions while preserving important features.

Dropout layers are introduced after certain MaxPooling2D layers to prevent overfitting by randomly setting some activations to zero during training. The feature maps are then flattened to a 1D array, which is processed by a Dense layer with 128 units and ReLU activation, followed by another BatchNormalization and Dropout layer. Finally, the output Dense layer consists of a single unit with a sigmoid activation function to perform binary classification between colon kras and colon non-kras cancer types. The model is trained and optimized using the Adam optimizer and binary cross-entropy loss function, with accuracy as the primary evaluation metric.

## Model Performance
Loss/Accuracy vs Epoch

![image](https://user-images.githubusercontent.com/116041838/229649913-45122685-a960-4fce-bdae-b76897bca4b9.png)


Confusion Matrix

![image](https://user-images.githubusercontent.com/116041838/229649991-045c83be-4a69-40f8-88f0-a1c70dd96ab5.png)


ROC-AUC curve

![image](https://user-images.githubusercontent.com/116041838/229650013-e8419e75-9355-4569-b904-d8f21558f247.png)

5 times trained result

<img width="610" alt="Screen Shot 2023-04-03 at 4 44 22 PM" src="https://user-images.githubusercontent.com/116041838/229650400-d3248ff2-56b5-4a96-bc1a-eee83cc5322b.png">


We have trained the model 5 times, obtaining an average accuracy of 0.82 and an ROC-AUC of 0.92. These results suggest that the model demonstrates a relatively strong performance in the classification task. It is essential to note that the average accuracy alone might not provide a comprehensive understanding of the model's performance, particularly in cases of class imbalance. The ROC-AUC score of 0.92 is a more informative metric, as it considers both sensitivity (true positive rate) and specificity (true negative rate). A high ROC-AUC score indicates that the model is effective at distinguishing between the classes while maintaining a low false positive rate. This is generally a desirable characteristic in classification tasks.

## Acknowledgements

