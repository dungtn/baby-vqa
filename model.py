from keras.models import Sequential
from keras.layers import Bidirectional, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Merge, Dropout, Flatten, Reshape
from keras.layers.convolutional import MaxPooling2D
from keras.layers.recurrent import LSTM, GRU


class LSTMModel(object):

    def __init__(self, vocab_size = 10000, img_dim=4096, word_dim=300, max_sent_len=26, nb_classes=1000, lstm_hidden_dim=512, fc_hidden_dim=2014, bidirect=True, dropout=0.5):
        self.vocab_size = vocab_size
        self.img_dim = img_dim
        self.word_dim = word_dim
        self.max_sent_len = max_sent_len
        self.nb_classes = nb_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.bidirect = bidirect
        self.dropout = dropout

    def build(self):
        self.img_model = Sequential()
        self.img_model.add(MaxPooling2D(input_shape=(14, 14, 512)))
        self.img_model.add(Flatten())
        for i in xrange(3):
            self.img_model.add(Dense(self.img_dim, activation='tanh'))
            self.img_model.add(BatchNormalization())

        self.txt_model = Sequential()
        # self.txt_model.add(Embedding(self.vocab_size, self.word_dim, input_length=self.max_sent_len, mask_zero = True))
        if self.bidirect:
            self.txt_model.add(Bidirectional(LSTM(output_dim=self.lstm_hidden_dim), input_shape=(self.max_sent_len, self.word_dim)))
            # self.txt_model.add(Bidirectional(GRU(output_dim=self.lstm_hidden_dim), input_shape=(self.max_sent_len, self.word_dim)))
        else:
            M = Masking(mask_value=0., input_shape=(self.max_sent_len, self.word_dim))
            self.txt_model.add(M)
            self.txt_model.add(LSTM(output_dim=self.lstm_hidden_dim, input_shape=(self.max_sent_len, self.word_dim)))
            # self.txt_model.add(GRU(output_dim=self.lstm_hidden_dim, input_shape=(self.max_sent_len, self.word_dim)))

        self.model = Sequential()
        self.model.add(Merge([self.txt_model, self.img_model], mode='concat', concat_axis=1))
        self.model.add(BatchNormalization())

        for i in xrange(2):
            self.model.add(Dense(self.fc_hidden_dim, init='he_normal', activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.nb_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()
        
    def fit(self, X_ques, X_img, y, nb_epoch=50, batch_size=50, shuffle=True):
        return self.model.fit([X_ques, X_img], y, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=shuffle)

    def evaluate(self, X_ques_test, X_im_test, y_test, batch_size=50):
    	return self.model.evaluate([X_ques_test, X_im_test], y_test, batch_size=batch_size)
    def train_on_batch(self, X_ques, X_img, y):
        return self.model.train_on_batch([X_ques, X_img], y)

	def save(self):
		params = {
			'img_dim': self.img_dim,
	        'word_dim': self.word_dim,
	        'max_sent_len': self.max_sent_len,
	        'nb_classes': self.nb_classes,
	        'lstm_hidden_dim': self.lstm_hidden_dim,
	        'fc_hidden_dim': self.fc_hidden_dim,
	        'bidirect': self.bidirect,
	        'dropout': self.dropout
		}
		fn = '../models/'+"".join(["{0}={1},".format(k,v) for k,v in params.iteritems()])
		open(fn+'.json', 'w').write(self.model.to_json())

