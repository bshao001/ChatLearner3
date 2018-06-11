# ChatLearner

![](https://img.shields.io/badge/python-3.6.2-brightgreen.svg)  ![](https://img.shields.io/badge/tensorflow-1.4.0-yellowgreen.svg?sanitize=true)

A chatbot implemented in TensorFlow based on the new sequence to sequence (NMT) model, with certain rules seamlessly integrated.

The core of ChatLearner (Papaya) was built on the NMT model(https://github.com/tensorflow/nmt), which has been adapted here to fit the needs of a chatbot. Due to the changes made on tf.data API in TensorFlow 1.4, this ChatLearner version only supports TF version 1.4 and later.

Before starting everything else, you may want to get a feeling of how ChatLearner behaves. Take a look at the sample conversation below or [here](https://github.com/bshao001/ChatLearner/blob/master/Data/Test/responses.txt), or if you prefer to try my trained model, download it [here](https://drive.google.com/file/d/1mVWFScBHFeA7oVxQzWb8QbKfTi3TToUr/view?usp=sharing). Unzip the downloaded .rar file, and copy the Result folder into the Data folder under your project root. A vocab.txt file is also included in case I update it without updating the trained model in the future.

![](/Data/Test/chat.png)

## Highlights and Specialties:
Why do you want to spend time checking this repository? Here are some possible reasons:

1. The Papaya Data Set for training the chatbot. You can easily find tons of training data online, but you cannot find any with such high quality. See the detailed description below about the data set.

2. The concise code style and clear implementation of the new seq2seq model based on dynamic RNN (a.k.a. the new NMT model). It is customized for chatbots and much easier to understand compared with the official tutorial.

3. The idea of using seamlessly integrated ChatSession to handle basic conversational context.

4. Some rules are integrated to demo how to combine traditional rule-based chatbots with new deep learning models. No matter how powerful a deep learning model can be, it cannot even answer questions requiring simple arithmetic calculations, and many others. The approach demonstrated here can be easily adapted to retrieve news or other online information. With the rules implemented, it can then properly answer many interesting questions. For example:
   
   * "What time is it now?" or "What day is it today?" or "What's the date yesterday?"
   * "Read me a story please." or "Tell me a joke." It can then present stories and jokes randomly and not being limited by the sequence length of the decoder.
   * "How much is twelve thousand three hundred four plus two hundred fifty six?" or "What is the sum of five and six?" or "How much is twelve thousand three-hundred and four divided by two-hundred-fifty-six?" or "If x=55 and y=19, how much is y - x?" or "How much do you get if you subtract eight from one hundred?" or even "If x = 99 and y = 228 / x, how much is y?"
   
   If you are not interested in rules, you can easily remove those lines related to knowledgebase.py and functiondata.py.
   
5. A SOAP-based web service (and a REST-API-based alternative, if you don't like to use SOAP) allows you to present the GUI in Java, while the model is trained and running in Python and TensorFlow.

6. A simple solution (in-graph) to convert a string tensor to lower case in TensorFlow. It is required if you utilize the new DataSet API (tf.data.TextLineDataSet) in TensorFlow to load training data from text files.

7. The repository also contains a chatbot implementation based on the legacy seq2seq model. In case you are interested in that, please check the Legacy_Chatbot branch at https://github.com/bshao001/ChatLearner/tree/Legacy_Chatbot.
   
## Papaya Conversational Data Set
Papaya Data Set is the best (cleanest and well-organized) free English conversational data you can find on the web for training a chatbot. Here are some details:

1. The data are composed of two sets: the first set was handcrafted, and we created the samples in order to maintain a consistent role of the chatbot, who can therefore be trained to be polite, patient, humorous, philosophical, and aware that he is a robot, but pretend to be a 9-year old boy named Papaya; the second set was cleaned from some online resources, including the scenario conversations designed for training robots, the Cornell movie dialogs, and cleaned Reddit data.

2. The training data set is split into three categories: two subsets will be augmented/repeated during the training, with different levels or times, while the third will not. The augmented subsets are to train the model with rules to follow, and some knowledge and common senses, while the third subset is just to help to train the language model.

3. The scenario conversations were extracted and reorganized from http://www.eslfast.com/robot/. If your model can support context, it would work much better by utilizing these conversations.

4. The original Cornell data set can be found at [here](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). We cleaned it using a Python script (the script can also be found in the Corpus folder); we then cleaned it manually by quickly searching certain patterns. 

5. For the Reddit data, a cleaned subset (about 110K pairs) is included in this repository. The vocab file and model parameters are created and adjusted based on all the included data files. In case you need a larger set, you can also find scripts to parse and clean the Reddit comments in the Corpus/RedditData folder. In order to use those scripts, you need to download a torrent of Reddit comments from a torrent link [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/). 
Normally a single month of comments is big enough (can generated 3M pairs of training samples roughly). You can tune the parameters in the scripts based on your needs. 

6. The data files in this data set were already preprocessed with NLTK tokenizer so that they are ready to feed into the model using new tf.data API in TensorFlow.

## Before You Proceed
1. Please make sure you have the correct TensorFlow version. It works only with TensorFlow 1.4, not any earlier releases because the tf.data API used here was newly updated in TF 1.4.

2. Please make sure you have environment variable PYTHONPATH setup. It needs to point to the project root directory, in which you have chatbot, Data, and webui folder. If you are running in an IDE, such as PyCharm, it will create that for you. But if you run any python scripts in a command line, you have to have that environment variable, otherwise, you get module import errors.

3. Please make sure you are using the same vocab.txt file for both training and inference/prediction. Keep in mind that your model will never see any words as we do. It's all integers in, integers out, while the words and their orders in vocab.txt help to map between the words and integers.

4. Spend a little bit time thinking of how big your model should be, what should be the maximum length of the encoder/decoder, the size of the vocabulary set, and how many pairs of the training data you want to use. Be advised that a model has a capacity limit: how much data it can learn or remember. When you have a fixed number of layers, number of units, type of RNN cell (such as GRU), and you decided the encoder/decoder length, it is mainly the vocabulary size that impacts your model's ability to learn, not the number of training samples. If you can manage not to let the vocabulary size to grow when you make use of more training data, it probably will work, but the reality is when you have more training samples, the vocabulary size also increases very quickly, and you may then notice your model cannot accommodate that size of data at all. Feel free to open an issue to discuss if you want.

## Training
Other than Python 3.6 (3.5 should work as well), Numpy, and TensorFlow 1.4. You also need NLTK (Natural Language Toolkit) version 3.2.4 (or 3.2.5).

During the training, I really suggest you to try playing with a parameter (colocate_gradients_with_ops) in function tf.gradients. You can find a line like this in modelcreator.py:
gradients = tf.gradients(self.train_loss, params). Set colocate_gradients_with_ops=True (adding it) and run the training for at least one epoch, note down the time, and then set
it to False (or just remove it) and run the training for at least one epoch and see if the times required for one epoch are significantly different. It is shocking to me at least.

Other than those, training is straightforward. Remember to create a folder named Result under the Data folder first. Then just run the following commands:

```bash
cd chatbot
python bottrainer.py
```

Good GPUs are highly recommended for the training as it can be very time-consuming. If you have multiple GPUs, the memory from all GPUs will be utilized by TensorFlow, and you can adjust the batch_size parameter in hparams.json file accordingly to make full use of the memory. You will be able to see the training results under Data/Result/ folder. Make sure the following 2 files exist as all these will be required for testing and prediction (the .meta file is optional as the inference model will be created independently): 

1. basic.data-00000-of-00001
2. basic.index

## Testing / Inference
For testing and prediction, we provide a simple command interface and a web-based interface. Note that vocab.txt file (and files in KnowledgeBase, for this chatbot) is also required for inference. In order to quickly check how the trained model performs, use the following command interface:

```bash
cd chatbot
python botui.py
```

Wait until you get the command prompt "> ".

A demo test result is provided as well. Please check it to see how this chatbot behaves now: https://github.com/bshao001/ChatLearner/blob/master/Data/Test/responses.txt

## Web Interface
A SOAP-based web service architecture is implemented, with a Python server and a Java client. A nice GUI is also included for your reference. For details, please check: https://github.com/bshao001/ChatLearner/tree/master/webui. Please be advised that certain information (such as pictures) is only available on the web interface (not in the command line interface).

A REST-API-based alternative is also given if SOAP is not your choice. For details, please check: https://github.com/bshao001/ChatLearner/tree/master/webui_alternative. Some of the latest updates may not be available with this option. Merge the changes from the other option if you need to use this.

![](/Data/Test/webui.png)

## References and Credits:
1. The new NMT model: https://github.com/tensorflow/nmt
2. Tornado Web Service: https://github.com/rancavil/tornado-webservices
3. Reddit data parser: https://github.com/pender/chatbot-rnn

# Other embeddings:
1. https://nlp.stanford.edu/projects/glove/ (https://nlp.stanford.edu/pubs/glove.pdf)
2. Pretrained word2vec from Google News (from gensim.models import Word2Vec)
3. Slimmed down Google News version in https://github.com/eyaler/word2vec-slim)

Embeddings can be pre-loaded in NMT and could help train the chatbot

