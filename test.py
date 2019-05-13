
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
import os
import model_RNN as rnnmodel
import utils
import csv
import time

with open("nowtime.txt", 'r') as f: # nowtime
    savepath = f.read()
sv_datetime = savepath

util = utils.Util()
MIN_SONG_LENGTH = 45
ms = time.strftime('_%H%M%S', time.localtime(time.time()))
filename = sv_datetime
filename = filename + ms

def test(trained_data, len_data, mode): # trained_data = songs_pitches or songs_durations, len_data = MIN_SONG_LENGTH, mode = pitch or duration
    # Test the RNN model
    #char2idx = util.getchar2idx(mode=mode)
    #trained_data = util.data2idx(trained_data, char2idx)
    rnn_model = rnnmodel.RNN(song_length=len_data,
                                   batch_size=1,
                                   mode='test')
    rnn_model.rnnNet(scopename=mode)

    # encoder input
    rnn_saver = tf.train.Saver(var_list=rnn_model.FC_vars)

    #np.random.seed(100)
    x_input = np.random.randn(1, rnn_model.sequence_length)

    with tf.Session() as sess:
        rnn_saver.restore(sess, "./save/" + sv_datetime + "/rnn_{}_model.ckpt".format(mode))
        prediction = sess.run(rnn_model.prediction, feed_dict={rnn_model.X: x_input})
        result = util.idx2char(prediction, mode)
        print("result : ", result)

        # print : result - trained_data
        print_error(result, trained_data, mode)

        return result

def print_error(result, trained_data, mode): # result : 생성된 곡, trained_data : 훈련데이터 100개
    # print : result - trained_data
    trained_data = trained_data[0] # 훈련데이터 중 맨 첫 데이터로 비교를 하려함.
    print("trained_data : ", trained_data)
    if mode == 'pitch':
        result_mean = []
        result_std = []
        cos_similarity = []
        pearson_correlation = []
        result = result[1:]
        trained_data = trained_data[1:]
        for i in range(0, len(result)):
            if result[i] == 'Rest':
                result[i] = np.mean(trained_data)
        error = [abs(int(x) - int(y)) for x, y in zip(result, trained_data)]
        cos_similarity.append(cos_sim(result, trained_data))
        result_mean.append(np.mean(result))
        result_std.append(np.std(result))
        pearson_correlation.append(pearson_cor(result, trained_data))
        print("error : ", error)
        print("total error : ", sum(error))
        print("cosine similarity : %0.3f" % cos_sim(result, trained_data))
        print("mean : %0.3f" % np.mean(result))
        print("std : %0.3f" % np.std(result))
        print("pearson correlation : %0.3f" % pearson_cor(result, trained_data))
        # 생성된 곡(result)의 mean과 std를 csv에 저장
        if (os.path.exists("./mean_std/mean_std.csv")) == False:
            with open('./mean_std/mean_std.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(result_mean)
                wr.writerow(result_std)
        else:
            with open('./mean_std/mean_std.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_mean)
                writer.writerow(result_std)
        # 생성된 곡의 cosine similarity와 pearson correlation 저장
        if (os.path.exists("./cos_sim/cos_sim_pearson.csv")) == False:
            with open('./cos_sim/cos_sim_pearson.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(cos_similarity)
                wr.writerow(pearson_correlation)
        else:
            with open('./cos_sim/cos_sim_pearson.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cos_similarity)
                writer.writerow(pearson_correlation)
    else:
        result_mean = []
        result_std = []
        cos_similarity = []
        pearson_correlation = []
        result = result[1:]
        trained_data = trained_data[1:]
        error = [abs(x - y) for x, y in zip(result, trained_data)]
        cos_similarity.append(cos_sim(result, trained_data))
        result_mean.append(np.mean(result))
        result_std.append(np.std(result))
        pearson_correlation.append(pearson_cor(result, trained_data))
        print("error : ", error)
        print("total error : ", sum(error))
        print("cosine similarity : %0.3f" % cos_sim(result, trained_data))
        print("mean : %0.3f" % np.mean(result))
        print("std : %0.3f" % np.std(result))
        print("pearson correlation : %0.3f" % pearson_cor(result, trained_data))


# 평가지표 1 : 코사인 유사도 (1에 가까우면 출력곡과 훈련데이터의 유사도가 높다는 것을 의미)
def cos_sim(result, trained_data):
    result = np.array(result)
    trained_data = np.array(trained_data)[:-1]
    cosine_similarity = np.dot(result, trained_data)/(np.linalg.norm(result)*np.linalg.norm(trained_data))
    return cosine_similarity
# 평가지표 2 : 피어슨 상관계수
def pearson_cor(result, trained_data):
    a = (len(result)-1) * np.std(result) * np.std(trained_data)
    b = 0
    for i in range(len(result)):
        b += (result[i]-np.mean(result))*(trained_data[i]-np.mean(trained_data))
    pearson_correlation = b/a
    return pearson_correlation

def main(_):
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
            # songs_pitches[0] - result 가 error

    pitches = test(songs_pitches, MIN_SONG_LENGTH, mode='pitch')
    durations = test(songs_durations, MIN_SONG_LENGTH, mode='duration')

    # make midi file
    util.song2midi(pitches, durations, '/generate', filename)


if __name__ == '__main__':
    tf.app.run()


