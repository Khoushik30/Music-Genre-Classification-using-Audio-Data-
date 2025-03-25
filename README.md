# Music Genre Classification using Deep Learning 🎵🎧

## 📌 Overview
This project focuses on classifying music genres using deep learning models like CNN, RNN, and a hybrid CNN-RNN architecture. We used the GTZAN Music Genre Dataset consisting of 10 genres and 1000 audio tracks.

Each model was trained on extracted MFCC features, and the hybrid model achieved the highest accuracy (88.4%) by combining spatial and temporal learning patterns from audio signals.

## 🧠 Technologies Used
- Python
- Librosa (Audio Feature Extraction)
- TensorFlow & Keras (Deep Learning Models)
- NumPy, Pandas (Data Handling)
- Matplotlib / Seaborn (Visualizations)
- Google Colab (Model Training & Testing)

## 🎶 Dataset
- **GTZAN Music Genre Dataset**
  - 1000 audio tracks (30 sec each)
  - 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
  - Features extracted: MFCC, Chroma, Spectral Contrast

## 🧪 Models Implemented

### 🔹 CNN (Convolutional Neural Network)
- Captures spatial features from spectrograms
- 2 convolution layers + pooling + dropout

### 🔹 RNN (LSTM - Recurrent Neural Network)
- Captures temporal patterns from audio sequences
- 2 LSTM layers + dense layers

### 🔹 Hybrid CNN-RNN
- Combines CNN for spatial features + LSTM for sequential features
- Achieved highest accuracy (~88.4%) on test set

All models were trained on extracted MFCC features for each audio clip.

## 📁 Project Structure

music-genre-classification/  
├── src/                          # Python source files  
│   ├── cnn_model.py  
│   ├── rnn_model.py  
│   ├── hybrid_cnn_rnn_model.py  
│   └── extract_features.py  
├── report/                       # Final project report  
│   └── music_genre_report.pdf  
├── README.md                     # Project documentation  

## 👨‍💻 Author

**Khoushik Raj Rasumalla**  
MS in Computer Science, Washington State University  
📫 Email: k.rasumalla@wsu.edu  
💼 LinkedIn: [linkedin.com/in/khoushikraj](https://www.linkedin.com/in/khoushikraj)  
💻 GitHub: [github.com/khoushikraj](https://github.com/khoushikraj)
