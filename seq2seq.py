import pickle
import random

import numpy as np
from keras import Input, Model, models
from keras.layers import Dense, Embedding, CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import dataloader


class Seq2Seq:
    def __init__(self):
        self._latent_dim = 256
        self._embedding_dim = 256
        self._epoch = 50
        self._encoder_tokenizer = None
        self._decoder_tokenizer = None
        self._model = None

    def _get_generator(self, encoder_input_data, decoder_input_data, decoder_output_data, batch_size=32):

        data_triplet = [encoder_input_data, decoder_input_data, decoder_output_data]
        begin_index = 0
        while True:
            batch_triple = data_triplet[begin_index:begin_index + batch_size]

            batch_encoder_input_data = np.array(batch_triple[0])
            batch_decoder_input_data = np.array(batch_triple[1])
            batch_decoder_output_data = np.array(batch_triple[2])

            yield [batch_encoder_input_data, batch_decoder_input_data], batch_decoder_output_data
            begin_index += batch_size
            if begin_index + batch_size > len(data_triplet):
                random.shuffle(batch_triple)
                begin_index = 0

    def predict(self, input_sentence):
        # Get encoder model
        _encoder_model = self._get_encoder_model()
        # Get decoder model
        _decoder_model = self._get_decoder_model()

        # Encode the input as state vectors.
        _input_seq = np.array(self._encoder_tokenizer.texts_to_sequences([input_sentence]))
        _states_value = _encoder_model.predict(_input_seq)

        # Build index to word dic
        _reverse_target_token_index = dict((idx, word) for word, idx in self._decoder_tokenizer.word_index.items())
        # Generate empty target sequence of length 1.
        target_seq = [self._decoder_tokenizer.word_index['<bos>']]

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            # Predict next token by previous word and previous states
            output_tokens, h, c = _decoder_model.predict([np.array(target_seq)] + _states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = _reverse_target_token_index[sampled_token_index]
            decoded_sentence += sampled_token + " "

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token == '<eos>' or
                    len(decoded_sentence) > 100):
                stop_condition = True

            target_seq = [self._decoder_tokenizer.word_index[sampled_token]]

            # Update states
            _states_value = [h, c]

        return decoded_sentence

    def training(self, filename, latent_dim=256, embedding_dim=256, epoch=50):

        self._latent_dim = latent_dim
        self._embedding_dim = embedding_dim
        self._epoch = epoch

        self._load_data(filename)
        self._build_tokenizer()
        self._build_network()
        self._training()

    def _load_data(self, filename):
        self._encoder_data, self._decoder_data = dataloader.load_data(filename)
        self._decoder_target_data = [x + " <EOS>" for x in self._decoder_data]
        self._decoder_data = ["<BOS> " + x for x in self._decoder_data]

        z = list(zip(self._encoder_data, self._decoder_data, self._decoder_target_data))
        random.shuffle(z)

        self._encoder_data[:], self._decoder_data[:], self._decoder_target_data[:] = zip(*z)

    def _build_tokenizer(self):
        self._build_encoder_tokenizer()
        self._build_decoder_tokenizer()

    def _build_encoder_tokenizer(self):
        self._encoder_tokenizer = Tokenizer(filters='', oov_token='oov', char_level=True)
        self._encoder_tokenizer.fit_on_texts(self._encoder_data)

    def _build_decoder_tokenizer(self):
        self._decoder_tokenizer = Tokenizer(filters='', oov_token='oov')
        self._decoder_tokenizer.fit_on_texts(self._decoder_data + self._decoder_target_data)

    def _build_network(self):
        self._build_encoder()
        self._build_decoder()
        self._model = Model(inputs=[self._encoder_inputs, self._decoder_inputs], outputs=self._decoder_outputs)

    def _build_encoder(self):
        num_encoder_tokens = len(self._encoder_tokenizer.word_index) + 1
        latent_dimension = self._latent_dim
        embedding_dimension = self._embedding_dim
        # layer init
        self._encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        self._encoder_embedding = Embedding(num_encoder_tokens, embedding_dimension, name="encoder_embedding")
        self._encoder = CuDNNLSTM(latent_dimension, return_state=True, name="encoder")

        encoder_outputs, encoder_state_h, encoder_state_c = self._encoder(self._encoder_embedding(self._encoder_inputs))
        self._encoder_states = [encoder_state_h, encoder_state_c]

    def _build_decoder(self):
        num_decoder_tokens = len(self._decoder_tokenizer.word_index) + 1
        latent_dimension = self._latent_dim
        embedding_dimension = self._embedding_dim

        # layer init
        self._decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        self._decoder_embedding = Embedding(num_decoder_tokens, embedding_dimension, name="decoder_embedding")
        self._decoder = CuDNNLSTM(latent_dimension, return_state=True, return_sequences=True, name="decoder")

        self._decoder_outputs, decoder_state_h, decoder_state_c = self._decoder(
            self._decoder_embedding(self._decoder_inputs), initial_state=self._encoder_states)

        self._decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
        self._decoder_outputs = self._decoder_dense(self._decoder_outputs)

    def _training(self):

        self._model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        _batch_size = 128
        _steps_per_epoch = len(self._encoder_data) // _batch_size
        _generator = self._get_generator_with_raw_format(_batch_size)

        self._model.fit_generator(generator=_generator, steps_per_epoch=_steps_per_epoch, epochs=self._epoch,
                                  shuffle=True, workers=4)

    def get_model(self):
        return self._model

    def _get_encoder_model(self):
        _encoder_inputs = self._model.get_layer(name="encoder_inputs").input
        _, encoder_state_h, encoder_state_c = self._model.get_layer(name="encoder").output
        encoder_states = [encoder_state_h, encoder_state_c]
        return Model(_encoder_inputs, encoder_states)

    def _get_decoder_model(self):
        # decoder state for each step
        _decoder_state_input_h = Input(shape=(None,), name="decoder_state_input_h")
        _decoder_state_input_c = Input(shape=(None,), name="decoder_state_input_c")
        decoder_states_inputs = [_decoder_state_input_h, _decoder_state_input_c]

        _decoder = self._model.get_layer(name="decoder")
        _decoder_inputs = self._model.get_layer(name="decoder_inputs").input
        _decoder_dense = self._model.get_layer(name="decoder_dense")
        _decoder_embedding = self._model.get_layer(name="decoder_embedding")

        decoder_outputs, state_h, state_c = _decoder(
            _decoder_embedding(_decoder_inputs), initial_state=decoder_states_inputs)
        decoder_states_outputs = [state_h, state_c]
        _decoder_outputs = _decoder_dense(decoder_outputs)
        return Model(
            [_decoder_inputs] + decoder_states_inputs,
            [_decoder_outputs] + decoder_states_outputs)

    def _get_generator_with_raw_format(self, batch_size=32):

        begin_index = 0
        while True:

            encoder_input_data = self._encoder_tokenizer.texts_to_sequences(
                self._encoder_data[begin_index:begin_index + batch_size])
            decoder_input_data = self._decoder_tokenizer.texts_to_sequences(
                self._decoder_data[begin_index:begin_index + batch_size])
            decoder_output_data = self._decoder_tokenizer.texts_to_sequences(
                self._decoder_target_data[begin_index:begin_index + batch_size])

            encoder_input_data = pad_sequences(encoder_input_data, padding="post")
            decoder_input_data = pad_sequences(decoder_input_data, padding="post")
            decoder_output_data = pad_sequences(decoder_output_data, padding="post")

            decoder_output_data = to_categorical(decoder_output_data, len(self._decoder_tokenizer.word_index) + 1)

            yield [encoder_input_data, decoder_input_data], decoder_output_data
            begin_index += batch_size
            if begin_index + batch_size > len(self._encoder_data):
                begin_index = 0
                continue

    def save(self, model_name):
        # saving
        with open('encoder_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self._encoder_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # saving
        with open('decoder_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self._decoder_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._model.save(model_name)

    def load(self, model_name):
        # loading
        with open('encoder_tokenizer.pickle', 'rb') as handle:
            self._encoder_tokenizer = pickle.load(handle)
        with open('decoder_tokenizer.pickle', 'rb') as handle:
            self._decoder_tokenizer = pickle.load(handle)
        self._model = models.load_model(model_name)
