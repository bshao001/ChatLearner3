# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf

from flask import Flask, request, jsonify
from settings import PROJECT_ROOT
from chatbot.botpredictor import BotPredictor

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
import math
import random
from numpy import mean


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
vocab = []
tsne = {}

DIGITS = 5
digits = math.pow(10,DIGITS)

@app.route('/reply', methods=['POST', 'GET'])
def reply():
    session_id = int(request.args.get('sessionId'))
    raw_question = request.args.get('question')
    blob_question = try_translate(raw_question) #UNICODE BUG ???

    if session_id not in predictor.session_data.session_dict:  # Including the case of 0
        session_id = predictor.session_data.add_session()

    answer, other_choices, probabilities = predictor.predict(session_id, blob_question.raw, html_format=True, fullResponse=True)
    answer = clean_string(answer)
    blob_answer = TextBlob(answer)
    polarity = combine_sentiments(blob_answer)
    word_descriptions=[]
    for n in range(len(other_choices)):
        word = vocab[other_choices[n][0][0]]
        if word != '_eos_' and word != 'rehashed':
            probability = float(int(digits*math.exp(probabilities[n][0][0]))/(digits*1.0))
            if word.lower() in tsne:
                x,y,pos = tsne[word.lower()]
            else:
                x,y,pos = 60*random.random(), 60*random.random(), random.randrange(21)
            word_descriptions.append( {"word": word, "prob": probability, "x": float(x), "y": float(y), "wordtype": int(pos) }  )
    return jsonify({'sessionId': session_id, 'sentence': answer, 'sentiment': polarity, 'words': word_descriptions})


@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
    sentence = request.args.get('sentence')
    blob = try_translate(sentence)
    word_descriptions = []
    for word in blob.words:
        if word.lower() in tsne:
            x,y,pos = tsne[word.lower()]
        else:
            x,y,pos = 60*random.random(), 60*random.random(), random.randrange(21)
        word_descriptions.append( {"word": word, "x": float(x), "y": float(y), "wordtype": int(pos) }  )

    return jsonify({'sentence': sentence, 'sentiment': blob.sentiment.polarity, 'words': word_descriptions})

@app.route('/combine_sentiment', methods=['POST', 'GET'])
def combine_sentiment():
    sentence = request.args.get('sentence')
    blob = try_translate(sentence)
    word_descriptions = []
    for word in blob.words:
        if word.lower() in tsne:
            x, y, pos = tsne[word.lower()]
        else:
            x, y, pos = 60 * random.random(), 60 * random.random(), random.randrange(21)
        word_descriptions.append({'word': word, 'x': float(x), 'y': float(y), 'wordtype': int(pos) })

    return jsonify({'sentence': sentence, 'sentiment': combine_sentiments(sentence), 'words': word_descriptions})

def try_translate(sentence):
    print (sentence)
    blob = TextBlob(sentence)
    lang = blob.detect_language()
    if lang == 'en':
        pass
    else:
        try:
            blob = blob.translate()
        except:
            pass
    return blob

def combine_sentiments(text, verbose=False):
    '''
    Combine Naive Bayes Classifier with Pattern Analyzer.
    '''
    blob_nb = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    blob_pa = TextBlob(text, analyzer=PatternAnalyzer())
    average_sentiment = mean([prob_to_polarity(blob_nb.sentiment.p_pos),
                            blob_pa.sentiment.polarity])
    return average_sentiment

def prob_to_polarity(prob):
    '''
    Convert probabilities [0,1] to polarities [-1, 1].
    '''
    return (prob - 0.5) * 2

def clean_string(answer, verbose=False):
    answer = str(answer)
    replace_list = ['"',  '´',  '`', '*']
    for strip_ in replace_list:
        answer = answer.replace(strip_, '' )
    if verbose:
        print(answer)
    return answer

if __name__ == "__main__":
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

    with open(os.path.join(corp_dir,"vocab.txt")) as fi:
        for line in fi:
            vocab.append(line.strip())
    with open(os.path.join(corp_dir,"1200K_glove_tSNE.csv")) as fi:
        fi.readline()
        for line in fi:
            sline = line.strip().split(",")
            tsne[sline[3]] = (sline[1],sline[2],sline[5])

    with tf.Session() as sess:
        predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
                                 result_dir=res_dir, result_file='basic')

        app.run(port=4567)
        print("Web service started.")
