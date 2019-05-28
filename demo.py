import os
import numpy as np
from midi import utils as midiutils

song_name, n_melody, pitches, durations = midiutils.midi2melody('./midi/demo/test.mid')
print("song name : ",song_name)
print("number of melody : ", n_melody)
print("pitches : ", pitches)
print("durations : ", durations)