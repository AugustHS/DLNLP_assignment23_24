import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

from model import  ModelA, ModelB
from transformers import BertTokenizer, BertForSequenceClassification
# ======================================================================================================================
# Data preprocessing
train_path = './Datasets/train.txt'
val_path = './Datasets/val.txt'
test_path = './Datasets/test.txt'

train_df = pd.read_csv(train_path, sep=';', names=['Text', 'Emotion'])
val_df = pd.read_csv(val_path, sep=';', names=['Text', 'Emotion'])
test_df = pd.read_csv(test_path, sep=';', names=['Text', 'Emotion'])

train_df = train_df.drop(train_df[(train_df['Emotion'] == 'surprise') | (train_df['Emotion'] == 'love')].index)
val_df = val_df.drop(val_df[(val_df['Emotion'] == 'surprise') | (val_df['Emotion'] == 'love')].index)
test_df = test_df.drop(test_df[(test_df['Emotion'] == 'surprise') | (test_df['Emotion'] == 'love')].index)

class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return features, labels

# data preprocessing for Model A
def preprocess_task_a(train_df, val_df, test_df):
    nltk.download('stopwords')
    vectorizer = CountVectorizer(max_features=5000, stop_words=stopwords.words('english'))
    train_feature = vectorizer.fit_transform(train_df['Text']).toarray()
    val_feature = vectorizer.transform(val_df['Text']).toarray()
    test_feature = vectorizer.transform(test_df['Text']).toarray()

    label_encoder = LabelEncoder()
    train_label = label_encoder.fit_transform(train_df['Emotion'])
    val_label = label_encoder.transform(val_df['Emotion'])
    test_label = label_encoder.transform(test_df['Emotion'])

    train_dataset = TextDataset(train_feature, train_label)
    val_dataset = TextDataset(val_feature, val_label)
    test_dataset = TextDataset(test_feature, test_label)

    return train_dataset, val_dataset, test_dataset, label_encoder

# data preprocessing for Model B
def preprocess_task_b(train_df, val_df, test_df):
    tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
    tokenizer.fit_on_texts(train_df['Text'])

    train_sequences = tokenizer.texts_to_sequences(train_df['Text'])
    val_sequences = tokenizer.texts_to_sequences(val_df['Text'])
    test_sequences = tokenizer.texts_to_sequences(test_df['Text'])

    max_sequence_length = 200
    train_feature = pad_sequences(train_sequences, maxlen=max_sequence_length, truncating='pre')
    val_feature = pad_sequences(val_sequences, maxlen=max_sequence_length, truncating='pre')
    test_feature = pad_sequences(test_sequences, maxlen=max_sequence_length, truncating='pre')

    label_encoder = LabelEncoder()
    train_label = label_encoder.fit_transform(train_df['Emotion'])
    val_label = label_encoder.transform(val_df['Emotion'])
    test_label = label_encoder.transform(test_df['Emotion'])

    train_dataset = TextDataset(train_feature, train_label)
    val_dataset = TextDataset(val_feature, val_label)
    test_dataset = TextDataset(test_feature, test_label)

    return train_dataset, val_dataset, test_dataset, label_encoder, tokenizer

# data processing for Model C
def preprocess_task_c(train_df, val_df, test_df,):

    train_data = train_df.copy()
    val_data = val_df.copy()
    test_data = test_df.copy()

    label_encoder = LabelEncoder()
    train_label = label_encoder.fit_transform(train_df['Emotion'])
    val_label = label_encoder.transform(val_df['Emotion'])
    test_label = label_encoder.transform(test_df['Emotion'])

    tokenizer = BertTokenizer.from_pretrained('./Model_C')

    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    def create_dataset(df, labels):
        texts = df['Text'].tolist()
        tokenized_data = tokenize_function(texts)
        encoded_labels = torch.tensor(labels)
        dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], encoded_labels)
        return dataset

    train_dataset = create_dataset(train_data, train_label)
    val_dataset = create_dataset(val_data, val_label)
    test_dataset = create_dataset(test_data, test_label)
    return train_dataset, val_dataset, test_dataset, label_encoder, tokenizer


train_dataset_a, val_dataset_a, test_dataset_a, label_encoder_a = preprocess_task_a(train_df, val_df, test_df)
train_loader_a = DataLoader(train_dataset_a, batch_size=64)
val_loader_a = DataLoader(val_dataset_a, batch_size=64)
test_loader_a = DataLoader(test_dataset_a, batch_size=64)

train_dataset_b, val_dataset_b, test_dataset_b, label_encoder_b, tokenizer_b = preprocess_task_b(train_df, val_df, test_df)
train_loader_b = DataLoader(train_dataset_b, batch_size=64)
val_loader_b = DataLoader(val_dataset_b, batch_size=64)
test_loader_b = DataLoader(test_dataset_b, batch_size=64)

train_dataset_c, val_dataset_c, test_dataset_c, label_encoder_c, tokenizer_c = preprocess_task_c(train_df, val_df, test_df)
train_loader_c = DataLoader(train_dataset_c, batch_size=64)
val_loader_c = DataLoader(val_dataset_c, batch_size=64)
test_loader_c = DataLoader(test_dataset_c, batch_size=64)

# ======================================================================================================================
# Task A(Model A) evaluation
input_dim_a = train_dataset_a.features.shape[1]
output_dim_a = len(label_encoder_a.classes_)

model_a = ModelA(input_dim_a, output_dim_a)
model_a.load_state_dict(torch.load('./Model_A.pth'))

criterion_a = nn.CrossEntropyLoss()
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001, weight_decay=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_a.to(device)

model_a.eval()
total_test_loss_a = 0
correct_predictions_a = 0
with torch.no_grad():
    for features, labels in test_loader_a:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model_a(features)
        loss = criterion_a(outputs, labels)
        total_test_loss_a += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions_a += torch.sum(predictions == labels)

avg_test_loss_a = total_test_loss_a / len(test_loader_a)
accuracy_a = correct_predictions_a.double() / len(test_loader_a.dataset)

# ======================================================================================================================
# Task A(ModelB) evaluation
vocab_size_b = 10000+ 1
embedding_dim_b = 200
hidden_dim_b = 256
output_dim_b = len(label_encoder_b.classes_)

model_b = ModelB(vocab_size_b, embedding_dim_b, hidden_dim_b, output_dim_b)
model_b.load_state_dict(torch.load('./Model_B.pth'))

criterion_b = nn.CrossEntropyLoss()
optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)
model_b.to(device)

model_b.eval()
total_test_loss_b = 0
correct_predictions_b = 0
with torch.no_grad():
    for features, labels in test_loader_b:
        features = features.to(device, dtype=torch.long)
        labels = labels.to(device)
        outputs = model_b(features)
        loss = criterion_b(outputs, labels)
        total_test_loss_b += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions_b += torch.sum(predictions == labels)

avg_test_loss_b = total_test_loss_b / len(test_loader_b)
accuracy_b = correct_predictions_b.double() / len(test_loader_b.dataset)

# ======================================================================================================================
# Task B(Model C) evaluation
model_c = BertForSequenceClassification.from_pretrained("./Model_C")

criterion_c = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_c.to(device)

model_c.eval()
total_test_loss_c = 0
correct_predictions_c = 0
with torch.no_grad():
    for batch in test_loader_c:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device, dtype=torch.long)

        outputs = model_c(input_ids, attention_mask=attention_mask)
        loss = criterion_c(outputs.logits, labels)
        total_test_loss_c += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct_predictions_c += torch.sum(predictions == labels)

avg_test_loss_c = total_test_loss_c / len(test_loader_c)
accuracy_c = correct_predictions_c.double() / len(test_loader_c.dataset)

# ======================================================================================================================
## Print out the results
print(f'Model A Test Loss: {avg_test_loss_a:.4f}')
print(f'Model A Test Accuracy: {accuracy_a:.4f}')
print(f'Model B Test Loss: {avg_test_loss_b:.4f}')
print(f'Model B Test Accuracy: {accuracy_b:.4f}')
print(f'Model C Test Loss: {avg_test_loss_c:.4f}')
print(f'Model C Test Accuracy: {accuracy_c:.4f}')