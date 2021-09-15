import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, Flatten, Input
from tensorflow.python.keras.layers import LSTM, Bidirectional, SimpleRNN, Conv1D, MaxPool1D
from keras.engine.topology import Layer, InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing import text as keras_text, sequence as keras_seq
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.utils import shuffle
from tensorflow import set_random_seed
import gc
import os

#   Data Cleaning
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re

#myrand=np.random.randint(1, 99999 + 1)
myrand=73849
np.random.seed(myrand)
set_random_seed(myrand)
z=0

EMBEDDING_SIZE=32
WORDS_SIZE=10000
INPUT_SIZE=1000
NUM_CLASSES=2
EPOCHS=10

#mydata = pd.read_csv('C:/Users/Ameer/Documents/emotion_model/Data/data_over_under.csv')
#mydata = shuffle(mydata)

mydata =  pd.read_csv('C:/Users/Ameer/Documents/emotion_model/Data/data_over_under.csv')
mydata1 =  pd.read_csv('C:/Users/Ameer/Documents/emotion_model/Data/data_over_under.csv')
mydata.append(mydata1)
mydata = shuffle(mydata)

mydata['text'] = mydata['text'].astype(str)
mydata['label'] = mydata['label'].astype(np.int64)

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

length = len(mydata)

nums = [0,length]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%1000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ))                                                                    
    clean_tweet_texts.append((mydata['text'][i]))

## Gabungkan balik dgn data
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['label'] = mydata.label
clean_df.head()

mydata = shuffle(clean_df)
mydata['text'] = mydata['text'].astype(str)

mydata['label'] = mydata['label'].astype(np.int64)

gc.collect()

mydata = shuffle(mydata)
mydata = shuffle(mydata)
mydata = shuffle(mydata)

#   Splitting the data into training (70%) and testing (30$) sets
x_train, x_test, y_train, y_test = train_test_split(mydata.iloc[:,0], mydata.iloc[:,1],
                                                    test_size=0.3, 
                                                    random_state=myrand, 
                                                    shuffle=True)
old_y_test = y_test

#Tokenize
tokenizer = keras_text.Tokenizer(char_level=False)
tokenizer.fit_on_texts(list(mydata['text']))
tokenizer.num_words=WORDS_SIZE

## Tokkenizing train data and create matrix
list_tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = keras_seq.pad_sequences(list_tokenized_train, 
                                  maxlen=INPUT_SIZE,
                                  padding='post')
x_train = x_train.astype(np.int64)

## Tokkenizing test data and create matrix
list_tokenized_test = tokenizer.texts_to_sequences(x_test)
x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                 maxlen=INPUT_SIZE,
                                 padding='post')
x_test = x_test.astype(np.int64)

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

#    Word2Seq CNN + Bi-RNN + Bi-LSTM model
model = Sequential(name='Word2Seq CNN + Bi-RNN + Bi-LSTM')

model.add(Embedding(input_dim =WORDS_SIZE,
                    output_dim=300,
                    input_length=INPUT_SIZE
                    ))
model.add(Conv1D(filters=300, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=15))
model.add(Conv1D(filters=300, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=10))
model.add(Bidirectional(SimpleRNN(300)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(7, activation='softmax'))

## Define multiple optional optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)

## Compile model with metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Word2Seq CNN + Bi-RNN + Bi-LSTM model built: ")
model.summary()

callbackdir= 'C:/Users/Ameer/Documents/emotion_model/ten'

tbCallback = TensorBoard(log_dir=callbackdir, 
                         histogram_freq=0, 
                         batch_size=128,
                         write_graph=True, 
                         write_grads=True, 
                         write_images=True)

tbCallback.set_model(model)

mld = 'C:/Users/Ameer/Documents/emotion_model/Models/word2seq_cnn_birnn_bilstm_new_37.hdf5'

## Create best model callback
mcp = ModelCheckpoint(filepath=mld, monitor="val_acc",
                      save_best_only=True, mode='max', period=1, verbose=1)

print('Training the Word2Seq CNN + Bi-RNN + Bi-LSTM model')
history = model.fit(x = x_train,
          y = y_train,
          validation_split = 0.3,
          epochs = EPOCHS,
          batch_size = 128,
          verbose =1,
          callbacks=[mcp,tbCallback])

print('\nPredicting the model')
model = load_model(mld)
results = model.evaluate(x_test, y_test, batch_size=128)
for num in range(0,2):
    print(model.metrics_names[num]+': '+str(results[num]))
    
##  Confusion Matrix
print('\nConfusion Matrix')
predicted = model.predict_classes(x_test)
confusion = confusion_matrix(y_true=old_y_test, y_pred=predicted)
print(confusion)

## Performance measure
print('\nWeighted Accuracy: '+ str(accuracy_score(y_true=old_y_test, y_pred=predicted)))
print('Weighted precision: '+ str(precision_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted recall: '+ str(recall_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted f-measure: '+ str(f1_score(y_true=old_y_test, y_pred=predicted, average='weighted')))


## Performance measure
print('Micro Precision: {:.2f}'.format(precision_score(y_true=old_y_test, y_pred=predicted, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_true=old_y_test, y_pred=predicted, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true=old_y_test, y_pred=predicted, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_true=old_y_test, y_pred=predicted, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_true=old_y_test, y_pred=predicted, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true=old_y_test, y_pred=predicted, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_true=old_y_test, y_pred=predicted, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true=old_y_test, y_pred=predicted, target_names=['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5', 'Class 6', 'Class 7']))


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()