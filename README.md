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

## Environment and tools
- Colab notebook
- NumPy
- Pandas
- Matplotlib and Seaborn
- scikit-image
- 

## EDA
