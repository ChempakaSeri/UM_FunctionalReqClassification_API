import flask
import pickle
import numpy as np
import pymysql
import os
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model
from flask import request, jsonify
import warnings

#   Import ARI
import textstat

global tokenizer
global pred_models 
global result
global INPUT_SIZE
global error

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['JSON_SORT_KEYS'] = False
warnings.filterwarnings('ignore')
tokenizer = None
error = None

pred_models = {}
INPUT_SIZE = {'word2seq_cnn':300,
              'word2vec_cnn':300,
              'word2seq_cnn_birnn':300,
              'word2vec_cnn_birnn':300,
              'word2vec_cnn_lstm':300,
              'word2vec_cnn_birnn_bilstm':300}

table_name = {'word2seq_cnn':'Word2Seq_CNN',
              'word2vec_cnn':'Word2Vec_CNN',
              'word2seq_cnn_birnn':'Word2Seq_CNN_BiRNN',
              'word2vec_cnn_birnn':'Word2Vec_CNN_BiRNN',
              'word2vec_cnn_lstm':'Word2Vec_CNN_LSTM',
              'word2vec_cnn_birnn_bilstm':'Word2Vec_CNN_BiRNN_BiLSTM'}

WORDS_SIZE = 10001

retName = ['Predicted requirement','Probability of the requirement is AI','Probability of the requirement is not AI']


## Main API get hook function
@app.route('/api/v1/req', methods=['GET'])
def api_req():
    global error
    error = False
    
    if 'text' in request.args:
        text = str(request.args['text'])
        if text == '':
            return "Error: No text provideed. Please specify a text."
        result = predict(text)
        return(jsonify(result))
    else:
        error = True
        return "Error: No text field provided. Please specify a text."

def predict(text):
    global pred_models
    return_dict={}
    return_list={}
    
    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences([text])
    return_list.update({'Text':text})
    
    for model in [model[:-5]for model in os.listdir('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model')]:
        x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                         maxlen=INPUT_SIZE[model],
                                         padding='post')
        x_test = x_test.astype(np.int64)

        ## Predict using the loaded model
        sentiment = 'The requirement is an AI' if pred_models[model].predict_classes(x_test)[0]==1 else 'The requirement is non-AI'
        positive_probability = pred_models[model].predict_proba(x_test)[0][1]
        negative_probability = pred_models[model].predict_proba(x_test)[0][0]
       
        return_dict.update({table_name[model].replace('_',' '): 
            {retName[0]:str(sentiment), 
             retName[1]:str(positive_probability), 
             retName[2]:str(negative_probability)}})
    
    return(return_dict)
def main():
    ## Load the Keras-Tensorflow models into a dictionary
    global pred_models 
    
    pred_models={'word2seq_cnn' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2seq_cnn.hdf5'),
                 'word2vec_cnn' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2vec_cnn.hdf5'),
                 'word2seq_cnn_birnn' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2seq_cnn_birnn.hdf5'),
                 'word2vec_cnn_birnn' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2vec_cnn_birnn.hdf5'),
                 'word2vec_cnn_lstm' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2vec_cnn_lstm.hdf5'),
                 'word2vec_cnn_birnn_bilstm' : load_model('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model/word2vec_cnn_birnn_bilstm.hdf5')}
    
    ## Make prediction function
    for model in [model[:-5]for model in os.listdir('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/Model')]:
        pred_models[model]._make_predict_function()
    
    ## Loading the Keras Tokenizer sequence file
    global tokenizer
    with open('C:/Users/Ameer/Documents/UM_FunctionalReqClassification_API/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    app.run(host='localhost', port=5020)

if __name__ == '__main__':
    main()


