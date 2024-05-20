# Text-based sentiment recognition
The purpose of this project is to solve the problem of text-based sentiment recognition using deep learning techniques in order to reach the goal of AI emotion care as an application. 
For this purpose, CNN, LSTM and fine-tuned pretrained BERT models were used to recognise and classify emotions in textual data and compared in terms of performance.
The Emotions Dataset for NLP downloaded from Kaggle was used as the original dataset for this project, and the performance of these models was comprehensively evaluated through operations such as data preprocessing, model construction, training and validation. 

## The organization of this project
There are three deep-learning models for this project, divided into two differnt tasks.
### Task A: Building two custom models (CNN and LSTM)
  - Model A: a CNN-based model
  - Model B: a LSTM-based model
### Task B: Fine-tuning a model (BERT)
  - Model C: a fine-tuned BERT model

Considering the amount of time it would take to run the full code for this project, **main.py** only contains the main part of my full code, and this file directly calls my well-trained models and outputs their accuracy on the test set.
So if you want to verify my results, you can just run **main.py** .
But if you want to go further into the full code of my project, please check out the jupyter notebook files in folders **A/** and **B/**.

## The role of each file
- **A/**: contains the full code for Task A
- **B/**:contains the full code for Task B
- **Datasets/**: contains the original datasets for this project
- **main.py**: contains the main code of this project
- **model.py**: contains the structure of Model A,B, and C
- **Model_A.pth**: contains the well-trained Model A
- **Model_B.pth**: contains the well-trained Model B
- **Model_C.pth**: contains the fine-tuned Model C

## Required packages
-**main.py**:
