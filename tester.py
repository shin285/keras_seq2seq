from seq2seq import Seq2Seq

seq2seq = Seq2Seq()
seq2seq.training(filename="sj2003.convert1.tag", epoch=200)
seq2seq.save("test_model")
seq2seq.load("test_model")

while True:
    text = input("input : ")  # Python 3
    if text == "END":
        break
    print("Your input is :"+text)
    predict_seq = seq2seq.predict(input_sentence=text)
    print(predict_seq)

print(seq2seq.predict("Hi this is JunsooShin"))

# model.summary()
#
# from keras.utils import plot_model
#
# plot_model(model, to_file='model.png', show_shapes=True)
