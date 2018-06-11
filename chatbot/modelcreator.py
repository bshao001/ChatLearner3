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
import tensorflow as tf
import chatbot.modelhelper as model_helper

from tensorflow.python.layers import core as layers_core


class ModelCreator(object):
    """Sequence-to-sequence model creator to create models for training or inference"""
    def __init__(self, training, tokenized_data, batch_input, scope=None):
        """
        Create the model.

        Args:
            training: A boolean value to indicate whether this model will be used for training.
            tokenized_data: The data object containing all information required for the model.
            scope: scope of the model.
        """
        self.training = training
        self.batch_input = batch_input
        self.vocab_table = tokenized_data.vocab_table
        self.vocab_size = tokenized_data.vocab_size
        self.reverse_vocab_table = tokenized_data.reverse_vocab_table

        hparams = tokenized_data.hparams
        self.hparams = hparams

        self.num_layers = hparams.num_layers
        self.time_major = hparams.time_major

        # Initializer
        initializer = model_helper.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.embedding = (model_helper.create_embbeding(vocab_size=self.vocab_size,
                                                        embed_size=hparams.num_units,
                                                        scope=scope))
        # This batch_size might vary among each batch instance due to the bucketing and/or reach
        # the end of the training set. Treat it as size_of_the_batch.
        self.batch_size = tf.size(self.batch_input.source_sequence_length)

        # Projection
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(
                    self.vocab_size, use_bias=False, name="output_projection")

        # Training or inference graph
        print("# Building graph for the model ...")
        res = self.build_graph(hparams, scope=scope)

        if training:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(self.batch_input.source_sequence_length) + \
                              tf.reduce_sum(self.batch_input.target_sequence_length)
            # Count the number of predicted words for compute perplexity.
            self.predict_count = tf.reduce_sum(self.batch_input.target_sequence_length)
        else:
            self.infer_logits, self.another_thing, self.final_context_state, self.sample_id = res
            self.sample_words = self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

        self.global_step = tf.Variable(0, trainable=False)

        params = tf.trainable_variables()

        # Gradients update operation for training the model.
        if training:
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            opt = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(self.train_loss, params)

            clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
                gradients, max_gradient_norm=hparams.max_gradient_norm)

            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("learning_rate", self.learning_rate),
                tf.summary.scalar("train_loss", self.train_loss),
            ] + gradient_norm_summary)
        else:
            self.infer_summary = tf.no_op()

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        if training:
            print("# Trainable variables:")
            for param in params:
                print("  {}, {}, {}".format(param.name, str(param.get_shape()), param.op.device))

    def train_step(self, sess, learning_rate):
        """Run one step of training."""
        assert self.training

        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size],
                        feed_dict={self.learning_rate: learning_rate})

    def build_graph(self, hparams, scope=None):
        """Creates a sequence-to-sequence model with dynamic RNN decoder API."""
        dtype = tf.float32

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Encoder
            encoder_outputs, encoder_state = self._build_encoder(hparams)

            # Decoder
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams)

            # Loss
            if self.training:
                loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    def _build_encoder(self, hparams):
        """Build an encoder."""
        source = self.batch_input.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, source)

            # Encoder_outpus: [max_time, batch_size, num_units]
            cell = self._build_encoder_cell(hparams)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                encoder_emb_inp,
                dtype=dtype,
                sequence_length=self.batch_input.source_sequence_length,
                time_major=self.time_major)

        return encoder_outputs, encoder_state

    def _build_encoder_cell(self, hparams):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return model_helper.create_rnn_cell(
            num_units=hparams.num_units,
            num_layers=hparams.num_layers,
            keep_prob=hparams.keep_prob)

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        """Build and run a RNN decoder with a final projection layer."""
        bos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.bos_token)), tf.int32)
        eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.eos_token)), tf.int32)

        # maximum_iteration: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.batch_input.source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state,
                self.batch_input.source_sequence_length)

            # Training
            if self.training:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = self.batch_input.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, target_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.batch_input.target_sequence_length,
                    time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state,)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
            # Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], bos_id)
                end_token = eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    # Helper
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding, start_tokens, end_token)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )
                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                if beam_width > 0:
                    logits = outputs #tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id
        return logits, sample_id, final_context_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        num_units = hparams.num_units
        num_layers = hparams.num_layers
        beam_width = hparams.beam_width

        dtype = tf.float32

        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        if not self.training and beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,
                                                                   multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                                          multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)

        cell = model_helper.create_rnn_cell(
            num_units=num_units,
            num_layers=num_layers,
            keep_prob=hparams.keep_prob)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (not self.training and beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            name="attention")

        if hparams.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.batch_input.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.batch_input.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        print (loss,logits)
        return loss

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def infer(self, sess):
        assert not self.training
        logits, infer_summary, _b, sample_words = sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
        ])
        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words, infer_summary, logits
