import tensorflow as tf
import numpy as np
import utils
import model_RNN as rnnmodel
from midi import utils as miditils

util = utils.Util()
sv_datetime = "20190425-2018"
MIN_SONG_LENGTH = 40

songs = util.get_all_song()
print("Load {} Songs...".format(len(songs)))
songs_name = ""
songs_len = []
songs_pitches = []
songs_durations = []
for song in songs:

    songs_name += "_" + song['name']
    songs_len.append(song['length'])
    songs_pitches.append(song['pitches'])
    songs_durations.append(song['durations'])

# 여러 곡의 길이를 제일 짧은 곡에 맞춘다.
for i in range(len(songs_pitches)):
    if len(songs_pitches[i]) > MIN_SONG_LENGTH: # min(songs_len)
        songs_pitches[i] = songs_pitches[i][:MIN_SONG_LENGTH]
        songs_durations[i] = songs_durations[i][:MIN_SONG_LENGTH]
print(songs_pitches)
print(songs_durations)
rnn_model = rnnmodel.RNN(song_length=MIN_SONG_LENGTH,
                                   batch_size=1,
                                   mode='test')
rnn_model.rnnNet(scopename='pitch')
char2idx = util.getchar2idx(mode='pitch')
trained_data = util.data2idx(songs_pitches, char2idx)
state_sample_data = util.data2idx(trained_data, char2idx)
x_sample_data, y_sample_data = rnn_model.data2idx(songs_pitches, mode='pitch')
print(x_sample_data)
print(np.shape(x_sample_data))



def test(trained_data, len_data, mode): # trained_data = songs_pitches or songs_durations, len_data = MIN_SONG_LENGTH
    # Test the RNN model, mode = pitch or duration
    char2idx = util.getchar2idx(mode=mode)

    rnn_model = rnnmodel.RNN(song_length=len_data,
                                   batch_size=1,
                                   mode='test')
    rnn_model.rnnNet(scopename=mode)

    # encoder input
    # x_sample_data, y_sample_data = rnn_model.data2idx(trained_data, mode=mode)
    trained_data = util.data2idx(songs_pitches, char2idx)
    # x_sample_data = np.reshape(x_sample_data, [rnn_model.batch_size, x_sample_data.shape[1]])  #
    # y_sample_data = np.reshape(y_sample_data, [rnn_model.batch_size, y_sample_data.shape[1]])
    rnn_saver = tf.train.Saver(var_list=rnn_model.FC_vars)
    np.random.seed(100)
    x_input = np.random.randn(1, MIN_SONG_LENGTH)
    with tf.Session() as sess:
        rnn_saver.restore(sess, "./save/" + sv_datetime + "/rnn_{}_model.ckpt".format(mode))

        prediction = sess.run(rnn_model.prediction, feed_dict={rnn_model.X: x_input})

        result = util.idx2char(prediction, mode)
        print("result : ", result)

        # print : result - trained_data
        print_error(result, trained_data, mode)

        return result
def print_error(result, trained_data, mode):
    # print : result - trained_data
    trained_data = trained_data[0]
    print("trained_data : ", trained_data)
    if mode == 'pitch':
        result = np.array(result)
        result['Rest' == result] = 0
        result = list(result)
        trained_data = np.array(trained_data)
        trained_data[trained_data == 'Rest'] = 0
        trained_data = list(trained_data)
        error = [abs(int(x) - int(y)) for x, y in zip(result, trained_data)]
        print("error : ", error)
        print("total error : ", sum(error))
    else:
        error = [abs(x - y) for x, y in zip(result, trained_data)]
        print("error : ", error)
        print("total error : ", sum(error))
# output song
pitches = test(songs_pitches, MIN_SONG_LENGTH, mode='pitch')
durations = test(songs_durations, MIN_SONG_LENGTH, mode='duration')

# make midi file
filename = sv_datetime

util.song2midi(pitches, durations, '/generate', filename)
