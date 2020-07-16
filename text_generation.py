from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.preprocessing.text import Tokenizer 
from keras.callbacks import EarlyStopping
import keras.utils as ku
import numpy as np

glove_path = 'Data/glove.twitter.27B/glove.twitter.27B.200d.txt'
tokenizer = Tokenizer()

def text_preprocess(text):

	# basic cleanup
	corpus = text.lower().split("\n")

	# tokenization	
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences 
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	# create predictors and label
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label, max_sequence_len, total_words

def rnn_model(predictors, label, max_sequence_len, total_words):
	
	model = Sequential()
	model.add(Embedding(total_words, 200, 
                     weights = [embedding_matrix],
                     input_length=max_sequence_len-1))
	model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(256)))
	model.add(Dense(total_words, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	model.fit(predictors, label, epochs=25, verbose=1, callbacks=[earlystop])
	return model 

def generate_text(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)	
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text


text = open('ham_lyrics.txt', encoding='latin1').read()

predictors, label, max_sequence_len, total_words = text_preprocess(text)

# GloVe embeddings
embeddings_index = dict()
with open(glove_path,
          encoding="utf8") as glove:
  for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  glove.close()

embedding_matrix = np.zeros((total_words, 200))
for word, index in tokenizer.word_index.items():
    if index > total_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

model = rnn_model(predictors, label, max_sequence_len, total_words)

model.save('hamilton_model.h5')

print("\n--- Spamilton ---")
print(generate_text("These United States", 3, max_sequence_len))
