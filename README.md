# EX-06 Named Entity Recognition
**DATE :**
### Aim:
To develop an LSTM-based model for recognizing the named entities in the text.

### Problem Statement and Dataset:
Named Entity Recognition (NER) is essential in NLP, focusing on identifying and classifying entities like persons, organizations, and locations in text. This project aims to develop an LSTM-based model to predict entity tags for each word in sentences. The dataset includes words, their POS tags, and standard NER tags such as ORGANIZATION, PERSON, LOCATION, and DATE. Accurate NER enhances applications like information extraction, question answering, and sentiment analysis.
### Design Steps:
- Step:1 Import necessary libraries like pandas, NumPy, and TensorFlow/Keras.   
- Step:2 Read the dataset and use forward fill to handle null values.
- Step:3 Create lists of unique words and tags, and count the number of unique entries
-n Step:4 Build dictionaries mapping words and tags to their corresponding index values.
- Step:5 Construct a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, and Time Distributed Dense layers, then compile it for training with the dataset.
### Program:
```python
#### Import Necessary Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from keras import layers
from keras.models import Model

##### Load and Preprocess Dataset

data = pd.read_csv("/content/ner_dataset.csv", encoding="latin1")
data.head()
data = data.fillna(method="ffill")
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
num_words = len(words)
tags=list(data['Tag'].unique())
num_tags = len(tags)
print("Unique tags are:", tags)

##### Define Sentence Getter Class

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func=lambda s:[(w,p,t)for w,p,t in zip(s["Word"].values.tolist(),
                        s["POS"].values.tolist(),s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

##### Create Index Dictionaries and Prepare Data

getter = SentenceGetter(data)
sentences = getter.sentences
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
X1 = [[word2idx[w[0]] for w in s] for s in sentences]

##### Pad Sequences and Split Data into Train/Test Sets

# Define max_len here
max_len = 100  # You can adjust this value as needed

X = sequence.pad_sequences(maxlen=max_len,sequences=X1, padding="post",value=num_words-1)
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
Y = sequence.pad_sequences(maxlen=max_len,sequences=y1,padding="post",value=tag2idx["O"])
Xtrain,Xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=1)

##### Build the LSTM Model Architecture

input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                  input_length=max_len)(input_word)
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)
bid_lstm = layers.Bidirectional(layers.LSTM(units=100,
              return_sequences=True,recurrent_dropout=0.1))(dropout)
output = layers.TimeDistributed(layers.Dense(num_tags,activation="softmax"))(bid_lstm)
model = Model(input_word, output)
model.summary()

##### Compile and Train the Model

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x=Xtrain, y=ytrain, validation_data=(Xtest,ytest),batch_size=50,epochs=3,)

##### Evaluate Model Performance and Plot Metrics

metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
plt.title('Training Accuracy vs. Validation Accuracy')
metrics[['loss','val_loss']].plot()
plt.title('Training Loss vs. Validation Loss')

##### Make Predictions and Display Results

i = 21
p = model.predict(np.array([Xtest[i]]))
p = np.argmax(p, axis=-1)
ytrue = ytest[i]
print("SANDHIYA R - 212222230129")
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(Xtest[i], ytrue, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

### Output:
![image](https://github.com/user-attachments/assets/052c5641-d617-4835-93a3-c90959d63a32)

![image](https://github.com/user-attachments/assets/b9caf7e4-2c48-4e43-b650-fb3c244be590)

![image](https://github.com/user-attachments/assets/0c30e6ca-90fb-4a60-b1d4-3e18982e0e2c)


### Result:
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
