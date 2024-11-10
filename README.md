# Spam Detection Project

## Overview

This project is a **spam detection application** built using **Streamlit** for the user interface, along with **NLTK** and **scikit-learn (sklearn)** for natural language processing and machine learning tasks. The dataset for this project was sourced from **Kaggle**. This tool allows users to input text and determine if it is spam or not, based on a trained machine learning model.

## Key Features

- **Streamlit Interface**: Provides an interactive and user-friendly web interface for real-time spam detection.
- **Text Processing**: Utilizes **NLTK** for text cleaning, tokenization, and feature extraction.
- **Machine Learning Model**: Built using **scikit-learn** to classify input text as spam or not.
- **Preprocessing Pipeline**: Includes data cleaning, vectorization (e.g., TF-IDF), and model training.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **NLTK (Natural Language Toolkit)**: For text preprocessing and tokenization.
- **scikit-learn (sklearn)**: For model building, training, and evaluation.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```

2. **Install the required packages**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Requirements

Ensure the following dependencies are in `requirements.txt`:
```
streamlit
nltk
scikit-learn
pandas
numpy
```

## Data Source

- The dataset used for training and testing was sourced from **Kaggle** and includes labeled messages to train the model on distinguishing spam from non-spam (ham).

## Conclusion

This **spam detection project** demonstrates a practical use of machine learning and NLP within an interactive web interface using **Streamlit**. It reflects the ability to integrate multiple technologies for developing functional applications with real-world use cases.
