# Define an input sequence and process it.
from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import dataloader
import numpy as np

latent_dim = 256

encoder_data, decoder_data = dataloader.load_data("sequence_data.test")
decoder_target_data = [x + " <EOS>" for x in decoder_data]
decoder_data = ["<BOS> " + x for x in decoder_data]

print(decoder_data+decoder_target_data)

encoder_tokenizer = Tokenizer(filters='', oov_token="<oov>")
encoder_tokenizer.fit_on_texts(encoder_data)

decoder_tokenizer = Tokenizer(filters='', oov_token="<oov>")
decoder_tokenizer.fit_on_texts(decoder_data+decoder_target_data)

num_encoder_tokens = len(encoder_tokenizer.word_index)
num_decoder_tokens = len(decoder_tokenizer.word_index)

encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

encoder_input_data = encoder_tokenizer.texts_to_sequences(encoder_data)
decoder_input_data = decoder_tokenizer.texts_to_sequences(decoder_data)
decoder_output_data = decoder_tokenizer.texts_to_sequences(decoder_target_data)

decoder_output_onehot = []
for decoder_indexes in decoder_output_data:
    sentence_onehot = []
    for index in decoder_indexes:
        sentence_onehot.append(to_categorical(index, num_decoder_tokens))
    decoder_output_onehot.append(sentence_onehot)

model.fit([np.array(encoder_input_data), np.array(decoder_input_data)], np.array(decoder_output_onehot),
          batch_size=128,
          epochs=10,
          validation_split=0.2)
