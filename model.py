from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM


class Model(object):

	def __init__(self, img_dim=4096, word_dim=300, max_sent_len=26, nb_classes=1000, lstm_hidden_dim=512, fc_hidden_dim=2014, bidirect=True, dropout=0.5):
		img_model = Sequential()
		img_model.add(Reshape(input_shape=(img_dim,), dims=(img_dim,)))

		txt_model = Sequential()
		# return sequences if use bidrectional LSTM
		txt_model.add(LSTM(output_dim=hidden_dim, return_sequences=bidirect, input_shape=(max_sent_len, word_dim)))
		#TODO: check if the paper actually use bi-lstm?!? not really understand what happened here!
		if bidirect:
			txt_model.add(LSTM(output_dim=hidden_dim, return_sequences=False))

		model = Sequential()
		model.add(Merge(txt_model, img_model), mode='concat', concat_axis=1)

		for i in xrange(2):
			model.add(Dense(fc_hidden_dim, init='uniform'))
			model.add(Activation('tanh'))
			model.add(Dropout(dropout))

		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
		return model

	def save(self):
		model_fn = '../models/'+"".join(["{0}={1},".format(k,v) for k,v in self.params.iteritems()])
		open(model_fn+'.json', 'w').write(model.to_json())