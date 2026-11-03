# Smart Text Predictor Using GRU

**NLP Mini Project - MCA (Artificial Intelligence & Machine Learning)**
Developed by: Aneesh Jaryal | Chandigarh University | UID 25MCI10117 | Class 25MAM1-B

---

## 1. Abstract
Smart Text Predictor is a text-prediction application built with two Gated Recurrent Unit (GRU) neural networks.
The first model (forward GRU) predicts the **next word** of an incomplete sentence. The second model
(reverse-context GRU) is trained on **reversed** sentences and predicts the **first word** of a sentence when the
user types only the remaining words. Both models are trained locally with TensorFlow/Keras on a small dataset about
AI, ML, deep learning, NLP, neural networks, Python, computer science and data science. A Streamlit web application
shows the top 3 or top 5 suggestions. No GPT API, paid API or external AI service is used.

## 2. Problem Statement
Typing is faster when a program can suggest words. Most simple predictors only look forward. This project builds a
predictor that can (a) continue a sentence and (b) recover the missing beginning of a sentence, using neural networks
that understand word order.

## 3. Objectives
1. Build a GRU model that predicts the next word and shows the top 3 / top 5 choices.
2. Build a separate reverse-context GRU model that predicts the first word.
3. Create a text-preprocessing pipeline (cleaning, tokenization, vocabulary, padding, splitting).
4. Evaluate the models with accuracy, loss and top-k accuracy on held-out data.
5. Provide an attractive Streamlit web interface.

## 4. Introduction to NLP
Natural Language Processing (NLP) is the branch of AI that helps computers understand and generate human language.
Computers only understand numbers, so NLP first converts words into numbers (tokenization + vocabulary), then uses
models to learn patterns. Typical NLP tasks: translation, sentiment analysis, chatbots, summarization and
**language modelling** (predicting the next word), which is what this project does.

## 5. What is a GRU?
A **Gated Recurrent Unit (GRU)** is a type of Recurrent Neural Network (RNN) made for sequences such as sentences.
It reads one word at a time and keeps a *hidden state* (a short memory). Two "gates" control the memory:

* **Update gate** - decides how much of the old memory to keep.
* **Reset gate** - decides how much of the old memory to forget when reading the new word.

Plain RNNs forget long-ago words (vanishing gradient problem). Gates solve this. A GRU is simpler and faster than an
LSTM because it has fewer gates and no separate cell state.

## 6. GRU Architecture used here
```
Input (a sequence of word IDs, padded to the same length)
   -> Embedding (64)   : each word ID becomes a vector of 64 numbers
   -> GRU (128 units)  : reads the sequence and summarises it
   -> Dropout (0.3)    : randomly switches off 30% of neurons while training (reduces overfitting)
   -> Dense (vocab)    : one output per word in the vocabulary
   -> Softmax          : turns the outputs into probabilities that add up to 1
```
Loss: `sparse_categorical_crossentropy` (the target is a word ID). Optimizer: **Adam** (learning rate 0.001).
Both models use the same architecture; they differ only in the **data** they are trained on.

## 7. Working Methodology
1. Read `dataset/sentences.txt` (and optional extra files).
2. Lowercase, remove punctuation, split into words (tokens).
3. Build a vocabulary (word -> ID). ID 0 = padding, ID 1 = unknown word.
4. Split **sentences** into train 80% / validation 10% / test 10%.
5. Create (input, target) samples, pad them with zeros at the front, and train the two GRUs.
6. Early stopping keeps the weights with the lowest validation loss.
7. Evaluate on the test sentences, save models, tokenizer, graphs and metrics.
8. The Streamlit app loads the saved files and predicts.

## 8. Next-Word Prediction Explained
Training sample creation for the sentence *"machine learning is a branch"*:

| Input words | Target (next word) |
|---|---|
| machine | learning |
| machine learning | is |
| machine learning is | a |
| machine learning is a | branch |

At prediction time the user's words are converted to IDs, padded, and given to the forward GRU. The softmax output
gives a probability for every word; we show the k highest (ignoring padding and unknown tokens).

## 9. First-Word Prediction Explained
A normal model cannot predict a *previous* word, so we **reverse** every sentence before training the second model:

`machine learning is a branch` -> reversed: `branch a is learning machine`

| Input (reversed) | Target |
|---|---|
| branch | a |
| branch a | is |
| branch a is | learning |
| branch a is learning | **machine** (the first word!) |

At prediction time the user types `is a branch` -> we reverse it to `branch a is` -> the reverse GRU predicts the word
that comes before it. This is a **separately trained model** (`first_word_gru.keras`); the next-word model is never
used backwards.
Optional filter: because the model is trained on every position, the app can limit suggestions to words that
*start sentences in the training data* (tick-box on the page, on by default). The probabilities are then rescaled
among those words. You can switch it off to see raw "preceding word" predictions.

## 10. Technologies Used
Python, TensorFlow/Keras (GRU), NumPy, Pandas (reading CSV/tables), Scikit-learn (train/test split), Matplotlib
(graphs), Streamlit (web app), Pickle (saving the vocabulary).

## 11. Dataset Details
`dataset/sentences.txt` - 240 simple sentences (30 per topic) on AI, ML, deep learning, NLP, neural networks,
Python, computer science and data science, one sentence per line.
**Your own data:** put sentences in a `.txt` file (one per line or normal paragraphs) or a `.csv` file (column named
`sentence`, else the first column) and run `python train.py --extra-data my_data.txt`.


**Larger dataset (recommended):** `dataset/sentences_large.txt` holds 3,000 extra pattern-based sentences about the same
topics (created by `dataset/generate_dataset.py`). Train with both files:
`python train.py --extra-data dataset/sentences_large.txt`.
Because these sentences follow fixed patterns, accuracy rises mainly for text written in a similar style; state this in
your report. Accuracy on completely free-form English will still be lower.

## 12. Project Structure
```
Smart_Text_Predictor_GRU/
    app.py              Streamlit web application (5 pages)
    train.py            trains both GRU models, evaluates, saves everything
    predict.py          loads models, predicts, terminal tool, evaluation
    requirements.txt    list of libraries
    README.md           this file
    dataset/sentences.txt   training sentences
    models/             next_word_gru.keras, first_word_gru.keras, tokenizer.pkl (created by train.py)
    utils/preprocessing.py  cleaning, tokenizing, vocabulary, samples, padding, splitting
    utils/model_builder.py  GRU architecture + accuracy helpers
    results/            training_history.png, metrics.json, model_summary.txt (created by train.py)
```

## 13. Installation (use Python 3.10, 3.11 or 3.12)
**Windows (PowerShell / Command Prompt)**
```
cd Smart_Text_Predictor_GRU
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
**Mac / Linux**
```
cd Smart_Text_Predictor_GRU
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 14. Execution
```
python train.py                 # 1) trains BOTH models (a few minutes on a normal CPU)
python predict.py --evaluate    # 2) evaluates the saved models on the test sentences
streamlit run app.py            # 3) opens the web app at http://localhost:8501
```
Terminal examples:
```
python predict.py --mode next  --text "machine learning is" --top-k 5
python predict.py --mode first --text "is a branch of artificial intelligence" --top-k 5
```
Optional: `python train.py --epochs 100`, `python train.py --extra-data my_data.txt`.

## 15. Results
Running `python train.py` writes your real numbers to `results/metrics.json` and the graph
`results/training_history.png`; the **Model Information** page of the app shows them in a table
(training/validation accuracy and loss, test top-1/top-3/top-5, parameters, true first-word accuracy).
Copy your own numbers here before submission:

| Metric | Forward GRU (next word) | Reverse GRU (first word) |
|---|---|---|
| Training accuracy | _fill from app_ | _fill from app_ |
| Validation accuracy | _fill from app_ | _fill from app_ |
| Test top-1 / top-3 | _fill from app_ | _fill from app_ |

## 16. Limitations
* **Small dataset** (240 sentences): the models memorise phrases of this dataset and may overfit; accuracy on new
  sentences is modest, and the vocabulary is only about 700 words.
* Many different words are valid after the same text, so top-1 accuracy is naturally low; top-3 / top-5 are fairer.
* Words outside the vocabulary are ignored.
* The first-word model must guess from the rest of the sentence only, which is a hard task (several first words fit).
* No pre-trained embeddings and no attention, so long-range meaning is limited.

## 17. Future Scope
Train on a large corpus (Wikipedia, news), use pre-trained embeddings (GloVe/Word2Vec), try LSTM / Bidirectional GRU /
Transformer models, add character-level or sub-word tokens, add beam search, and deploy online.

## 18. Conclusion
The project shows that two small GRU networks, one trained forward and one trained on reversed text, can provide
useful next-word and first-word suggestions locally, with a simple and attractive Streamlit interface. With more data
the same code can give much better predictions.

---

# What every file does (beginner guide)
| File | Job |
|---|---|
| `train.py` | The "teacher". Loads data, prepares it, trains Model 1 and Model 2, tests them, saves files. Run once. |
| `predict.py` | The "brain user". Loads the saved models and answers questions. `Predictor.predict_next` = Feature A, `Predictor.predict_first` = Feature B. |
| `app.py` | The web page. Calls `predict.py`. Contains 5 pages, the CSS styling and the buttons. |
| `utils/preprocessing.py` | Turns text into numbers (clean -> tokens -> IDs -> padded arrays). |
| `utils/model_builder.py` | Defines the GRU network and the accuracy calculations. |
| `models/tokenizer.pkl` | The saved vocabulary (word <-> ID) and settings. Needed at prediction time. |

## Important code blocks explained
* `clean_text()` - lowercase + remove punctuation so "Machine," and "machine" are the same word.
* `build_vocab()` - gives each word a number. Neural networks need numbers, not words.
* `make_samples(reverse=...)` - creates the (input -> target) pairs. `reverse=True` flips each sentence first;
  **this one line is the difference between the two models**.
* `pad_sequences_pre()` - inputs have different lengths; padding with zeros makes them equal so they fit in a matrix.
* `build_gru_model()` - Embedding -> GRU -> Dropout -> Dense(softmax).
* `EarlyStopping` in `train.py` - stops training when validation loss stops improving, and keeps the best weights.
* `Predictor.predict_first()` - reverses the user's words, asks the reverse GRU, optionally filters to sentence-starting words.
* `st.session_state` in `app.py` - Streamlit forgets variables on every click; session state remembers the typed text and results.

## How to explain the project in your viva (60-second script)
"My project is Smart Text Predictor using GRU. I trained two GRU networks in TensorFlow. The first reads a sentence
left to right and predicts the next word. The second is trained on reversed sentences, so it can predict the word that
comes before a given text - that gives first-word prediction. I clean the text, build a vocabulary, convert words to
IDs, pad them, and train with Adam and cross-entropy loss. I evaluated with accuracy and top-k accuracy on held-out
sentences. The Streamlit app shows the top 3 or 5 suggestions. The dataset is small, so results are limited, but
more data would improve them."

---

# Viva Questions and Answers
1. **What is NLP?** The field of AI that helps computers understand and generate human language.
2. **What is a language model?** A model that gives the probability of the next word given the previous words.
3. **What is an RNN?** A neural network for sequences that keeps a hidden state (memory) while reading word by word.
4. **What is a GRU?** A gated type of RNN with an update gate and a reset gate that controls what to remember and forget.
5. **GRU vs LSTM?** GRU has fewer gates and parameters, trains faster; LSTM has a separate cell state and three gates.
6. **Why not a plain RNN?** It forgets long-ago words because of the vanishing-gradient problem.
7. **What does the update gate do?** Decides how much old information to keep.
8. **What does the reset gate do?** Decides how much old information to forget when reading a new word.
9. **What is tokenization?** Splitting text into words (tokens).
10. **What is a vocabulary?** The list of unique words with a number (ID) for each.
11. **Why do we convert words to numbers?** Neural networks work only with numbers.
12. **What is padding?** Adding zeros so all inputs have the same length.
13. **What is an Embedding layer?** It converts each word ID into a small vector of numbers that the model learns; similar words get similar vectors.
14. **What is Dropout?** Randomly switching off neurons during training to reduce overfitting.
15. **Why Softmax in the last layer?** It converts outputs into probabilities that sum to 1, one per word.
16. **Which loss did you use and why?** Sparse categorical cross-entropy, because the target is one word ID among many classes.
17. **Why Adam optimizer?** It adapts the learning rate automatically and usually converges quickly.
18. **How does next-word prediction work?** Words -> IDs -> pad -> forward GRU -> softmax -> show the top-k words.
19. **How does first-word prediction work?** Reverse the typed words, give them to the reverse-context GRU, which predicts the word that comes before them.
20. **Why a separate model for first-word prediction?** A forward model learned "what comes after"; it cannot predict "what comes before" unless trained on reversed order.
21. **What is overfitting? How did you reduce it?** The model memorises training data. We used Dropout, validation data and early stopping.
22. **What is the validation set for?** To watch performance on unseen data during training and to stop early.
23. **Why split by sentence?** So the same sentence is never in both training and test data.
24. **What is top-k accuracy?** The prediction is correct if the true word is among the k best guesses.
25. **Why is top-1 accuracy low?** Many words can be correct after the same text, and the dataset is small.
26. **What are the limitations?** Small dataset, limited vocabulary, ignores unknown words, no pre-trained embeddings.
27. **How can you improve it?** More data, pre-trained embeddings, Bidirectional GRU/LSTM/Transformers, larger vocabulary.
28. **What is an epoch? Batch size?** One full pass over the training data; the number of samples used per weight update.
29. **What does Pickle do here?** Saves the vocabulary and settings (`tokenizer.pkl`) so the app can use them later.
30. **Why Streamlit?** It builds a web interface using only Python.
"# Text-predictor" 
