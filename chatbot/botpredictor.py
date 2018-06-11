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
import nltk
import os
import string
import tensorflow as tf

from chatbot.tokenizeddata import TokenizedData
from chatbot.modelcreator import ModelCreator
from chatbot.knowledgebase import KnowledgeBase
from chatbot.sessiondata import SessionData
from chatbot.patternutils import check_patterns_and_replace
from chatbot.functiondata import call_function

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BotPredictor(object):
    def __init__(self, session, corpus_dir, knbase_dir, result_dir, result_file):
        """
        Args:
            session: The TensorFlow session.
            corpus_dir: Name of the folder storing corpus files and vocab information.
            knbase_dir: Name of the folder storing data files for the knowledge base.
            result_dir: The folder containing the trained result files.
            result_file: The file name of the trained model.
        """
        self.session = session

        # Prepare data and hyper parameters
        print("# Prepare dataset placeholder and hyper parameters ...")
        tokenized_data = TokenizedData(corpus_dir=corpus_dir, training=False)

        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.load_knbase(knbase_dir)

        self.session_data = SessionData()

        self.hparams = tokenized_data.hparams
        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src_dataset = tf.data.Dataset.from_tensor_slices(self.src_placeholder)
        self.infer_batch = tokenized_data.get_inference_batch(src_dataset)

        # Create model
        print("# Creating inference model ...")
        self.model = ModelCreator(training=False, tokenized_data=tokenized_data,
                                  batch_input=self.infer_batch)
        # Restore model weights
        print("# Restoring model weights ...")
        self.model.saver.restore(session, os.path.join(result_dir, result_file))

        self.session.run(tf.tables_initializer())

    def predict(self, session_id, question, html_format=False, fullResponse=False):
        chat_session = self.session_data.get_session(session_id)
        chat_session.before_prediction()  # Reset before each prediction

        if question.strip() == '':
            answer = "Don't you want to say something to me?"
            chat_session.after_prediction(question, answer)
            return answer

        pat_matched, new_sentence, para_list = check_patterns_and_replace(question)

        for pre_time in range(2):
            tokens = nltk.word_tokenize(new_sentence.lower())
            tmp_sentence = [' '.join(tokens[:]).strip()]

            self.session.run(self.infer_batch.initializer,
                             feed_dict={self.src_placeholder: tmp_sentence})

            outputs, _ , logits = self.model.infer(self.session)
            if self.hparams.beam_width > 0:
                outputs = outputs[0]

            eos_token = self.hparams.eos_token.encode("utf-8")
            outputs = outputs.tolist()[0]

            if eos_token in outputs:
                outputs = outputs[:outputs.index(eos_token)]

            if pat_matched and pre_time == 0:
                out_sentence, if_func_val = self._get_final_output(outputs, chat_session,
                                                                   para_list=para_list,
                                                                   html_format=html_format)
                if if_func_val:
                    chat_session.after_prediction(question, out_sentence)
                    return out_sentence
                    if fullResponse:
                        return out_sentence,logits[0],logits[1][0]
                    else:
                        return out_sentence
                else:
                    new_sentence = question
            else:
                out_sentence, _ = self._get_final_output(outputs, chat_session,
                                                         html_format=html_format)
                chat_session.after_prediction(question, out_sentence)
                if fullResponse:
                    return out_sentence,logits[0],logits[1][0]
                else:
                    return out_sentence

    def _get_final_output(self, sentence, chat_session, para_list=None, html_format=False):
        sentence = b' '.join(sentence).decode('utf-8')
        if sentence == '':
            return "I don't know what to say.", False

        if_func_val = False
        last_word = None
        word_list = []
        for word in sentence.split(' '):
            word = word.strip()
            if not word:
                continue

            if word.startswith('_func_val_'):
                if_func_val = True
                word = call_function(word[10:], knowledge_base=self.knowledge_base,
                                     chat_session=chat_session, para_list=para_list,
                                     html_format=html_format)
                if word is None or word == '':
                    continue
            else:
                if word in self.knowledge_base.upper_words:
                    word = self.knowledge_base.upper_words[word]

                if (last_word is None or last_word in ['.', '!', '?']) and not word[0].isupper():
                    word = word.capitalize()

            if not word.startswith('\'') and word != 'n\'t' \
                and (word[0] not in string.punctuation or word in ['(', '[', '{', '``', '$']) \
                and last_word not in ['(', '[', '{', '``', '$']:
                word = ' ' + word

            word_list.append(word)
            last_word = word

        return ''.join(word_list).strip(), if_func_val
