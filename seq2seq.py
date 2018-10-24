from keras import Input, Model
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer

import numpy as np
import dataloader


class Seq2Seq:
    def __init__(self):
        pass

    def training(self, filename):
        self._build_tokenizer(filename)
        self._build_network()
        self._training()

    def _build_tokenizer(self, filename):
        self._encoder_data, self._decoder_data = dataloader.load_data(filename)
        self._decoder_target_data = [x + " <EOS>" for x in self._decoder_data]
        self._decoder_data = ["<BOS> " + x + " <EOS>" for x in self._decoder_data]

        self._build_encoder_tokenizer()
        self._build_decoder_tokenizer()

    def _build_encoder_tokenizer(self):
        self._encoder_tokenizer = Tokenizer(filters='', oov_token="<oov>")
        self._encoder_tokenizer.fit_on_texts(self._encoder_data)

    def _build_decoder_tokenizer(self):
        self._decoder_tokenizer = Tokenizer(filters='', oov_token="<oov>")
        self._decoder_tokenizer.fit_on_texts(self._decoder_data)

    def _build_network(self):
        self._build_encoder()
        self._build_decoder()
        self._model = Model(inputs=[self._encoder_inputs, self._decoder_inputs], outputs=self._decoder_outputs)

    def _build_encoder(self):
        num_encoder_tokens = len(self._encoder_tokenizer.word_index)
        latent_dimension = 128
        # layer init
        self._encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        self._encoder_embedding = Embedding(num_encoder_tokens, 256, name="encoder_embedding")
        self._encoder = LSTM(latent_dimension, return_state=True, name="encoder")

        encoder_outputs, encoder_state_h, encoder_state_c = self._encoder(self._encoder_embedding(self._encoder_inputs))
        self._encoder_states = [encoder_state_h, encoder_state_c]

    def _build_decoder(self):
        num_decoder_tokens = len(self._decoder_tokenizer.word_index)
        latent_dimension = 128
        # layer init
        self._decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        self._decoder_embedding = Embedding(num_decoder_tokens, 256, name="decoder_embedding")
        self._decoder = LSTM(latent_dimension, return_state=True, return_sequences=True, name="decoder")

        self._decoder_outputs, decoder_state_h, decoder_state_c = self._decoder(
            self._decoder_embedding(self._decoder_inputs), initial_state=self._encoder_states)

        self._decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="dense")
        self._decoder_outputs = self._decoder_dense(self._decoder_outputs)

    def _training(self):
        encoder_input_data = self._encoder_tokenizer.texts_to_sequences(self._encoder_data)
        decoder_input_data = self._decoder_tokenizer.texts_to_sequences(self._decoder_data)
        decoder_output_data = self._decoder_tokenizer.texts_to_sequences(self._decoder_target_data)

        self._model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self._model.fit([np.array(encoder_input_data), np.array(decoder_input_data)], np.array(decoder_output_data),
                        batch_size=32,
                        epochs=10,
                        validation_split=0.2)

    def get_model(self):
        return self._model
