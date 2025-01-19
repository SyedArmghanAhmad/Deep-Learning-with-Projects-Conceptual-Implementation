
# Fake News Classifier 📚🤖  

A deep learning project aimed at combating the spread of misinformation by classifying news as real or fake using advanced NLP techniques and Bidirectional LSTM networks.  

---

## 🧐 Problem Statement  

Fake news is a growing problem in today’s digital era, where misinformation spreads rapidly across platforms. This project aims to provide a reliable solution by leveraging machine learning and natural language processing to classify news articles effectively.  

---

## 🚀 Approach  

1. **Data Preprocessing**:  
   - Removed missing values.  
   - Cleaned text (lowercased, removed special characters, stopwords, etc.).  
   - Applied stemming to reduce words to their root forms.  

2. **Text Encoding & Representation**:  
   - Transformed text using One-Hot Encoding.  
   - Applied padding to standardize input lengths.  
   - Embedded the text into dense semantic vectors using an Embedding layer.  

3. **Model Architecture**:  
   - Built a **Bidirectional LSTM** model to capture forward and backward context in the text.  
   - Added a dense layer with sigmoid activation for binary classification.  

4. **Evaluation Metrics**:  
   - Accuracy, precision, recall, and F1-score to assess the model’s performance.  

---

## 🔑 Why Bidirectional LSTM?  

Bidirectional LSTMs allow the model to learn contextual relationships from both directions in a sequence. For example, understanding a word might require both its preceding and succeeding words in a sentence.  

### Code Snippet  

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Input(shape=(sent_length,)),  # Define input shape
    Embedding(input_dim=voc_size, output_dim=embedding_vector_features),
    Bidirectional(LSTM(100)),  # Captures forward and backward text dependencies
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

### Advantages of Bidirectional LSTM  

- Captures richer contextual understanding of text.  
- Improves classification performance, especially in language-related tasks.  

---

## 🎯 Results  

| Metric         | Value   |  
|----------------|---------|  
| **Accuracy**   | 91.88%  |  
| **Precision**  | 88%     |  
| **Recall**     | 94%     |  
| **F1-Score**   | 91%     |  

The model demonstrates robust performance in distinguishing between fake and real news articles.  

---

## 🔧 Installation  

1. Clone the repository:  

   ```bash
   git clone https://github.com/yourusername/FakeNewsClassifier.git
   cd FakeNewsClassifier
   ```  

2. Install dependencies:  

   ```bash
   pip install -r requirements.txt
   ```  

3. Run the notebook:  
   - Open `FakeNewsClassifier.ipynb` in Google Colab or your local Jupyter environment.  

---

## 📈 Future Enhancements  

- Implement transformer models like BERT for improved contextual understanding.  
- Integrate pretrained embeddings like GloVe for semantic-rich word representations.  
- Build a web app interface for real-time classification.  

---

## 🤝 Contributing  

Contributions are welcome! Please fork the repository and create a pull request with your enhancements or fixes.  

---

## 📜 License  

This project is licensed under the MIT License.  

---

## 🙌 Acknowledgements  

- [TensorFlow](https://www.tensorflow.org/)  
- [NLTK](https://www.nltk.org/)  
- [Keras Sequential Model](https://keras.io/guides/sequential_model/)  

``` bash

### Key Improvements in this Version:
1. **Engaging Structure**: Organized sections with a clean hierarchy for readability.  
2. **Conciseness**: Explained concepts without overloading with unnecessary details.  
3. **Focus on Impact**: Highlighted the use of Bidirectional LSTM as a core innovation.  
4. **User-Friendly Steps**: Clear setup and execution instructions.  
5. **Professional Tone**: Polished language with emphasis on the project's significance.  

Let me know if you'd like further refinements! 😊
