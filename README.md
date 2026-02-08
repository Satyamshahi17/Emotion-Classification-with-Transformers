# Emotion Classification with Transformers

## Project Overview

This project implements two distinct approaches for emotion classification using the Hugging Face Transformers library and the Emotion dataset. The goal is to classify text into one of six emotion categories: sadness, joy, love, anger, fear, and surprise.

The project demonstrates the trade-offs between computational efficiency and model performance by comparing feature extraction with a classical machine learning classifier versus end-to-end fine-tuning of a transformer model.

**Dataset**: Emotion Dataset from Hugging Face
- 6 emotion classes: sadness, joy, love, anger, fear, surprise
- Training samples: 16,000
- Validation samples: 2,000
- Test samples: 2,000

---

## Approach 1: Feature Extraction with Logistic Regression

### Methodology

This approach uses DistilBERT as a frozen feature extractor. The pre-trained model generates contextualized embeddings for each text sample, and these embeddings are used to train a simple Logistic Regression classifier.

**Steps:**
1. Load the pre-trained `distilbert-base-uncased` model
2. Tokenize the input text
3. Extract hidden states (768-dimensional embeddings from the [CLS] token)
4. Train a Logistic Regression classifier on the extracted features

### Implementation

```python
from transformers import TFAutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression

# Load pre-trained model
model_name = 'distilbert-base-uncased'
model = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract hidden states
def extract_hidden_states(batch):
    inputs = {k:v for k,v in batch.items() if k in tokenizer.model_input_names}
    last_hidden_state = model(**inputs).last_hidden_state
    return {'hidden_state': last_hidden_state[:,0]}

emotions_hidden_data = emotions_encoded.map(extract_hidden_states, batched=True)

# Train classifier
lr = LogisticRegression(max_iter=3000)
lr.fit(X_train, y_train)
```

### Results

- **Validation Accuracy**: 63%
- **Training Time**: Fast (minutes on CPU)
- **Computational Cost**: Low

### Advantages and Limitations

**Advantages:**
- Fast training time
- Low computational requirements (CPU sufficient)
- Good for quick prototyping and baseline models
- Minimal infrastructure needed

**Limitations:**
- Lower accuracy compared to fine-tuning
- Model weights are not adapted to the specific task
- Fixed representations may not capture task-specific patterns

---

## Approach 2: End-to-End Fine-Tuning

### Methodology

This approach fine-tunes the entire DistilBERT model for the emotion classification task. All model parameters are updated during training, allowing the model to adapt its representations specifically for emotion detection.

**Steps:**
1. Load `TFAutoModelForSequenceClassification` with 6 output classes
2. Configure optimizer with weight decay and learning rate scheduling
3. Compile and train

### Implementation

```python
from transformers import TFAutoModelForSequenceClassification, create_optimizer

# Load model for sequence classification
num_labels = 6
tf_model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels
)

# Create optimizer with weight decay
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=len(train_dataset) * 3,
    weight_decay_rate=0.01,
)

# Compile and train
tf_model.compile(optimizer=optimizer, metrics=['accuracy'])
tf_model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=2)
```

### Results

- **Validation Accuracy**: 94%
- **Training Time**: 2-3 hours on GPU
- **Computational Cost**: Moderate (GPU recommended)

**Training Progress:**
```
Epoch 1/2: loss: 0.6667 - accuracy: 0.7721 - val_loss: 0.1962 - val_accuracy: 0.9290
Epoch 2/2: loss: 0.1544 - accuracy: 0.9389 - val_loss: 0.1361 - val_accuracy: 0.9405
```

### Advantages and Limitations

**Advantages:**
- High accuracy (94% validation accuracy)
- Model adapts to task-specific patterns
- State-of-the-art performance
- Learns task-specific representations

**Limitations:**
- Requires more computational resources (GPU recommended)
- Longer training time
- Risk of overfitting on small datasets
- Higher infrastructure costs

---

## Comparison of Approaches

| Metric | Approach 1 (Feature Extraction) | Approach 2 (Fine-Tuning) |
|--------|--------------------------------|--------------------------|
| **Validation Accuracy** | 63% | 94% |
| **Training Time** | Minutes | Hours |
| **Compute Required** | CPU sufficient | GPU recommended |
| **Parameters Trained** | ~4,700 (LR only) | ~67M (entire model) |
| **Best Use Case** | Quick baseline, limited resources | Production deployment, maximum accuracy |

The fine-tuning approach achieves a 31% improvement in accuracy over feature extraction, demonstrating the significant value of task-specific adaptation despite the additional computational cost.

---

## Tech Stack

**Libraries and Frameworks:**
- `transformers` - Hugging Face Transformers library for pre-trained models
- `datasets` - Hugging Face Datasets for data loading
- `tensorflow` - Deep learning framework
- `tf-keras` - Keras API for TensorFlow
- `scikit-learn` - Machine learning library for Logistic Regression and metrics
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `umap-learn` - Dimensionality reduction for visualization

**Model:**
- DistilBERT-base-uncased (66M parameters, 6 transformer layers, 768 hidden size)

**Development Environment:**
- Python 3.x
- Jupyter Notebook

---

## Installation

```bash
pip install transformers
pip install datasets
pip install tensorflow
pip install tf-keras
pip install scikit-learn
pip install umap-learn
pip install sentencepiece
pip install numpy pandas matplotlib
```

---

## Usage

### Load Dataset
```python
from datasets import load_dataset
emotion_dataset = load_dataset('emotion')
```

### Approach 1: Feature Extraction
Run the notebook cells for hidden state extraction and Logistic Regression training.

### Approach 2: Fine-Tuning
Run the notebook cells for TFAutoModelForSequenceClassification training.

---

## Key Findings

1. **Transfer Learning Effectiveness**: Pre-trained language models provide strong starting points for downstream tasks
2. **Performance vs Efficiency Trade-off**: Feature extraction offers quick results but fine-tuning provides significantly better performance
3. **Task-Specific Adaptation**: Fine-tuning yields a 31% accuracy improvement, demonstrating the importance of adapting model weights to the specific task
4. **Resource Considerations**: The choice between approaches should be based on available computational resources and accuracy requirements

---

## Future Work

- Experiment with other transformer models (BERT, RoBERTa, ALBERT)
- Implement early stopping and model checkpointing
- Explore ensemble methods
- Create web interface for real-time predictions

---

## Author

**Satyam Kumar**  
Computer Science and Engineering Undergraduate

---
