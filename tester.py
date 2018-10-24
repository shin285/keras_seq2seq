import dataloader
import seq2seq


def dataloader_test(filename):
    return dataloader.load_data(filename)


dataloader_test("sequence_data.test")

seq2seq = seq2seq.Seq2Seq()
seq2seq.training(filename="sequence_data.test")
model = seq2seq.get_model()

model.summary()
encoder_layer = model.get_layer("encoder")

from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True)

