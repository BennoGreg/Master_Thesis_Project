import shelve
import librosa.display
import matplotlib.pyplot as plt
import re
import os.path



def plot_time_domain(data, title, basepath=None, persist=False, frame=None):
    if frame != None:
        title = "%s_%d" % (title, frame)

    librosa.display.waveshow(data, sr=16000, color='b', )
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    if persist:
        plt.savefig("%s/%s" % (basepath, title), dpi=200)
    else:
        plt.show()
    plt.close()

def extract_instrument_info(label):
        m = re.search("(([0-9]{3}-?){3})_([0-9])", label)
        instrument_pitch = m.group(1)
        part = m.group(3)
        return instrument_pitch, part

def get_instruments_of_one_pitch(instruments, pitch):
    reg = r"-(" + re.escape(pitch) + ")-"
    instruments = filter(lambda instrument: re.search(reg, instrument), instruments)

    return list(instruments)


# these functions return the hardcoded time and frequency arrays according to the current STFT configuration.
def get_time_resolution(nfft_half, times, sf):
    times_array = []
    for i in range(1, times+1):
        times_array.append(nfft_half/sf*i)

    return times_array

def get_frequency_bins(sf, hop_length):

    freqs = []
    freq_res = sf/2/hop_length
    for i in range(0, hop_length+1):
        freqs.append(freq_res*i)

    return freqs

def check_path(path):
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)
