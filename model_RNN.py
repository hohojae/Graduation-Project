
# RNN model Class
import utils
import tensorflow as tf
import numpy as np
tf.set_random_seed(77)
layers = tf.contrib.layers
util = utils.Util()
class RNN(object):
    def __init__(self,
                 song_length,
                 batch_size=2,
                 mode='train'):

        # mode 'train' or 'test'
        self.mode = mode

        # set index <-> data4
        # melody [0, 1 ~ 36, 50] -- size 38
        # rhythm [0, 1, 2, 3, 4, 6, 8, 12, 16] -- size 9
        self.melody_sample = list(range(1, 37)) # melody_sample에 [1,2,3,..., 35,36] 리스트 대입
        self.melody_sample.append(50)  # rest   # melody_sample=[1,2,...,35,36,50]
        self.melody_sample.append(0)  # for test  melody_sample=[1,2,...,36,50,0] -- size 38
        self.idx2char = list(set(self.melody_sample))  # melody_sample을 집합으로 만든 후 list로 변환해 인덱스로 참조가능, idx2char의 인덱스로 멜로디 찾을 수 있음
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)} # char2idx = {0: 0, 1: 1, 2: 2, 3: 3, ... , 35: 35, 36: 36, 50: 37} 키=데이터, 아이템=인덱스, 멜로디를 인덱스로

        # set hyperparameter
        self.input_size = 48  # 38(=멜로디의 개수와 같음) len(self.char2idx)
        self.hidden_size = 128                 # 셀에서의 출력 크기
        self.output_size = 48  # 38
        self.batch_size = batch_size
        self.sequence_length = song_length - 1 # 송렝스 아직 모름, 시퀀스 렝스=송렝스-1
        self.learning_rate = 0.01

        # placeholder(데이터타입=int32, shape, 해당 placeholder의 이름 "x_data", "y_data"
        self.X = tf.placeholder(tf.int32, [None, self.sequence_length], name='x_data')
        self.Y = tf.placeholder(tf.int32, [None, self.sequence_length], name='y_data')
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,  # 히든사이즈=128
                                             state_is_tuple=True)
    def data2idx(self, data, mode):
        x_data = [] # x_data 초기화
        y_data = [] # y_data 초기화
        for d in data:
            train_data = []
            train_data = [util.getchar2idx(mode)[i] for i in d[:]] # data의 멜로디를 인덱스로 바꾼다.
            x_data.append(train_data[:-1])
            y_data.append(train_data[1:])


        x_data = np.array(x_data) # x_data를 배열로 변환 (적은 메모리로 데이터를 빠르게 처리가능)
        y_data = np.array(y_data) # y_data를 배열로, 리스트는 속도가 느리고 메모리 많이 차지함.
        return x_data, y_data # 배열조건 : 모든원소가 같은 자료형이어야 한다. 원소의 개수를 바꿀수없다.
    def rnnNet(self, scopename):
        self.rnn_scope = "rnn_{}".format(scopename)

        self.x_one_hot = tf.one_hot(self.X, self.input_size) # X는 None * sequence_length이다. input_size = 46
                                                                # 여기서 시퀀스렝스는 송렝스-1인데 송렝스는 40으로 임의로 설정하였다.
        with tf.variable_scope(self.rnn_scope):

            self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)  # 셀 모양 그대로 0을 채워놓은 텐서를 저장해놓는다.


            if (self.mode == 'train'):
                self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
                #self.initial_state = tf.random_normal([self.batch_size, self.hidden_size], mean=0.0, stddev=1.0,
                #                                      dtype=tf.float32)
            elif (self.mode == 'test'):
                self.initial_state = self.cell.zero_state(1, dtype=tf.float32)
                #self.initial_state = tf.random_normal([self.batch_size, self.hidden_size], mean=0.0, stddev=1.0,
                 #                                     dtype=tf.float32)

            ########### dynamic ##########
            outputs, state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.x_one_hot, initial_state=self.initial_state)

            # output size : [batch_size, seqence_length, hidden_size]
            X_for_fc = tf.reshape(outputs, [-1, self.hidden_size]) # hidden_size = 128
            outputs = layers.fully_connected(inputs=X_for_fc,
                                             num_outputs=self.output_size,
                                             activation_fn=None)
            self.outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_size])
            self.prediction = tf.argmax(self.outputs, axis=2)

        self.FC_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.rnn_scope)  # RNN 스코프에서 트레이너블 배리어블을 불러모음.

    def build(self, scopename):
        print("Start model build...")
        self.loss_scope = "loss_{}".format(scopename)
        # one_hot : [x_data, input_size(=38)]를 원핫인코딩 한것

        self.rnnNet(scopename)

        with tf.variable_scope(self.loss_scope): # loss스코프
            weights = tf.ones([self.batch_size, self.sequence_length]) # 가중치 적용, 배치사이즈 X 시퀀스렝스
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, # Weighted cross-entropy loss for a sequence of logits.
                                                             targets=self.Y,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.FC_vars)
            tf.summary.scalar("loss", self.loss)
        self.loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.loss_scope)

        # summary for tensorboard graph
        self._create_summaries()

        print('complete model build.')

    def _create_summaries(self):
        with tf.variable_scope('summaries'):
            summ_loss = tf.summary.scalar(self.loss_scope, self.loss)

            self.summary_op = tf.summary.merge([summ_loss])