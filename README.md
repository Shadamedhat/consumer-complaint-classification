# Consumer Complaint Classification

Classify customer complaint narratives into product categories using deep learning and NLP, then deploy the best model as an interactive web app with Gradio.

## Dataset

Consumer Complaints dataset (processed version) from Kaggle: [add your Kaggle link here]

- Input: complaint narrative (text)
- Target: product category (5 classes)
  - `credit_reporting`
  - `debt_collection`
  - `mortgages_and_loans`
  - `credit_card`
  - `retail_banking`

## Class distribution and imbalance

The original data was highly imbalanced:

| Class | Complaints |
|-------|-----------|
| credit_reporting | 91,172 |
| debt_collection | 23,148 |
| mortgages_and_loans | 18,990 |
| credit_card | 15,566 |
| retail_banking | 13,535 |

To handle the imbalance I capped each class at 7,000 samples (undersampling the large classes) and also used class weights during training. The train/test split is stratified (80/20), so every class has the same proportion in both sets.

## Pipeline

1. Text cleaning (lowercase, remove links, numbers and punctuation)
2. Tokenization with the Keras `Tokenizer` (10,000 words + OOV token)
3. Padding to a fixed length of 200 tokens
4. Embedding layer (100 dimensions) to represent words as dense vectors
5. Training four models:
   - SimpleRNN
   - LSTM
   - GRU
   - Fine-tuned DistilBERT (HuggingFace Transformers)
6. Evaluation with Accuracy, Precision, Recall, F1-score and Confusion Matrix
7. Comparison of the four models and prediction on new complaints
8. Deployment with Gradio

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| **DistilBERT** | **0.8593** | **0.8592** | **0.8593** | **0.8591** |
| LSTM | 0.8183 | 0.8195 | 0.8183 | 0.8177 |
| GRU | 0.8110 | 0.8124 | 0.8110 | 0.8112 |
| SimpleRNN | 0.2237 | 0.2435 | 0.2237 | 0.1938 |

Best model: **DistilBERT** (F1 = 0.859).

## Analysis

- DistilBERT is the best model. It is pre-trained on a large amount of text and uses attention, so it understands the context of each word better than models trained from scratch.
- LSTM and GRU perform similarly (about 0.81 to 0.82 F1). Both use gates that let them remember information across long complaints. GRU has fewer parameters, so it trains faster.
- SimpleRNN performed close to random guessing (with 5 balanced classes, random is about 20%). SimpleRNN struggles with long sequences because of the vanishing gradient problem, and with 200-token inputs most of its memory is lost before the end of the text.

## Demo

The Gradio app takes a complaint narrative and returns the predicted category with a confidence score.

## Tools

Python, TensorFlow/Keras, HuggingFace Transformers, PyTorch, scikit-learn, pandas, matplotlib, seaborn, Gradio

## Files

- `consumer_complaint_classification.ipynb`: full notebook (EDA, preprocessing, training, evaluation, deployment)
