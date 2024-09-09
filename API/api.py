from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import os
import pandas as pd
model = tf.keras.models.load_model('item_model.h5')
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))
X = df['comment_text']
vectorizer.adapt(X.values)


app = Flask(__name__)

@app.route('/api', methods = ['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    input_str = vectorizer(inputchr)
    res = model.predict(np.expand_dims(input_str,0))
    answer = res[0][0]
    answer1=answer*100
    answer2=1-answer1
    if answer2<0.1:
        answer2=answer2*10
    # answer = str(ord(inputchr))
    d['output'] = str(answer2)
    print(answer2)
    return d



if __name__ =="__main__":
    app.run()
