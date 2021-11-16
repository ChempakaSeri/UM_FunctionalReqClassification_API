import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, Flatten
from tensorflow.python.keras.layers import Conv1D, MaxPool1D
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing import text as keras_text, sequence as keras_seq
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
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
myrand=58584
np.random.seed(myrand)
set_random_seed(myrand)
z=0

EMBEDDING_SIZE=32
WORDS_SIZE=10000
INPUT_SIZE=300
NUM_CLASSES=2
EPOCHS=100

# To allow dynamic GPU memory allowcation for model training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  
config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

#   Read Data
mydata =  pd.read_csv('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/balanced_data.csv')
mydata = shuffle(mydata)

mydata['text'] = mydata['text'].astype(str)
mydata['label'] = mydata['label'].astype(np.int64)
mydata['label'].value_counts().head(2)

#   Data Cleaning
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


nums = [0,len(mydata)]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%100 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ))                                                                    
    clean_tweet_texts.append(tweet_cleaner(mydata['text'][i]))

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

#   Split dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(mydata.iloc[:,0], mydata.iloc[:,1],
                                                    test_size=0.3, 
                                                    random_state=myrand, 
                                                    shuffle=True)

old_y_test = y_test

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

y_train = to_categorical(y_train, num_classes=NUM_CLASSES).astype(np.int64)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES).astype(np.int64)

#   Model Creation
model = Sequential(name='Word2Seq CNN')

model.add(Embedding(input_dim =WORDS_SIZE,
                    output_dim=300,
                    input_length=INPUT_SIZE
                    ))
model.add(Conv1D(filters=300, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=15))
model.add(Conv1D(filters=300, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=10))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(2, activation='softmax'))


## Define multiple optional optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)

## Compile model with metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )
print("Word2Seq CNN model built: ")
model.summary()

## Create TensorBoard callbacks

callbackdir= 'C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/ten'

tbCallback = TensorBoard(log_dir=callbackdir, 
                         histogram_freq=0, 
                         batch_size=128,
                         write_graph=True, 
                         write_grads=True, 
                         write_images=True)

tbCallback.set_model(model)

mld = 'C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/word2seq_cnn_balanced.hdf5'

## Create best model callback
mcp = ModelCheckpoint(filepath=mld, monitor="val_acc",
                      save_best_only=True, mode='max', period=1, verbose=1)

print('Training the Word2Seq CNN model')
history = model.fit(x = x_train,
          y = y_train,
          validation_split = 0.3,
          epochs = EPOCHS,
          batch_size = 128,
          verbose = 1,
          callbacks=[mcp,tbCallback])

print('\nPredicting the model')
model = load_model(mld)
results = model.evaluate(x_test, y_test, batch_size=128)
for num in range(0,2):
    print(model.metrics_names[num]+': '+str(results[num]))

print('\nConfusion Matrix')
predicted = model.predict_classes(x_test)
confusion = confusion_matrix(y_true=old_y_test, y_pred=predicted)
print(confusion)

## Performance measure
print('\nWeighted Accuracy: '+ str(accuracy_score(y_true=old_y_test, y_pred=predicted)))
print('Weighted precision: '+ str(precision_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted recall: '+ str(recall_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted f-measure: '+ str(f1_score(y_true=old_y_test, y_pred=predicted, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true=old_y_test, y_pred=predicted, target_names=['Class 1', 'Class 2']))



acc_1 = history.history['acc']
val_acc_1 = history.history['val_acc']
loss_1 = history.history['loss']
val_loss_1 = history.history['val_loss']

epochs_range_1 = range(len(acc_1))

plt.plot(epochs_range_1, acc_1, 'bo', label='Training acc')
plt.plot(epochs_range_1, val_acc_1, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()



plt.plot(epochs_range_1, loss_1, 'bo', label='Training loss')
plt.plot(epochs_range_1, val_loss_1, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


## <----    GRAPH PLOT    ---->
plt.plot(epochs_range_1, acc_1, 'b', label='Word2Seq CNN', color='blue')
plt.plot(epochs_range_2, acc_2, 'b', label='Word2Seq CNN BiRNN', color='red')
plt.plot(epochs_range_3, acc_3, 'b', label='Word2Vec CNN', color='yellow')
plt.plot(epochs_range_4, acc_4, 'b', label='Word2Vec CNN BiRNN', color='green')
plt.plot(epochs_range_5, acc_5, 'b', label='Word2Vec CNN BiRNN BiLSTM', color='purple')
plt.plot(epochs_range_6, acc_6, 'b', label='Word2Vec CNN BiLSTM', color='orange')
plt.title('Training Accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs_range_1, val_acc_1, 'b', label='Word2Seq CNN', color='blue')
plt.plot(epochs_range_2, val_acc_2, 'b', label='Word2Seq CNN BiRNN', color='red')
plt.plot(epochs_range_3, val_acc_3, 'b', label='Word2Vec CNN', color='yellow')
plt.plot(epochs_range_4, val_acc_4, 'b', label='Word2Vec CNN BiRNN', color='green')
plt.plot(epochs_range_5, val_acc_5, 'b', label='Word2Vec CNN BiRNN BiLSTM', color='purple')
plt.plot(epochs_range_6, val_acc_6, 'b', label='Word2Vec CNN BiLSTM', color='orange')
plt.title('Validation Accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs_range_1, loss_1, 'b', label='Word2Seq CNN', color='blue')
plt.plot(epochs_range_2, loss_2, 'b', label='Word2Seq CNN BiRNN', color='red')
plt.plot(epochs_range_3, loss_3, 'b', label='Word2Vec CNN', color='yellow')
plt.plot(epochs_range_4, loss_4, 'b', label='Word2Vec CNN BiRNN', color='green')
plt.plot(epochs_range_5, loss_5, 'b', label='Word2Vec CNN BiRNN BiLSTM', color='purple')
plt.plot(epochs_range_6, loss_6, 'b', label='Word2Vec CNN BiLSTM', color='orange')
plt.title('Training Loss')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs_range_1, val_loss_1, 'b', label='Word2Seq CNN', color='blue')
plt.plot(epochs_range_2, val_loss_2, 'b', label='Word2Seq CNN BiRNN', color='red')
plt.plot(epochs_range_3, val_loss_3, 'b', label='Word2Vec CNN', color='yellow')
plt.plot(epochs_range_4, val_loss_4, 'b', label='Word2Vec CNN BiRNN', color='green')
plt.plot(epochs_range_5, val_loss_5, 'b', label='Word2Vec CNN BiRNN BiLSTM', color='purple')
plt.plot(epochs_range_6, val_loss_6, 'b', label='Word2Vec CNN BiLSTM', color='orange')
plt.title('Validation Loss')
plt.legend()
plt.figure()
plt.show()