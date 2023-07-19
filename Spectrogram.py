import gc
import os.path
import re
import shelve
import numpy as np

from utils import check_path, get_instruments_of_one_pitch


class Spectrogram:

    def __init__(self, samplename, instrument, frequencies, times, magnitude, phase, ref_power):
        self.name = samplename
        self.instrument = instrument
        self.frequencies = frequencies
        self.times = times
        self.magnitude = magnitude
        self.phase = phase
        self.ref_power = ref_power

    def persist(self, dataset):
        path = "./SpectrogramData.nosync/%s_mel/%s" % (dataset, self.instrument)
        check_path(path)
        s = shelve.open("%s/%s" % (path, self.name), "c")
        s['spec'] = self
        s.close()

    def plot_spec(self, savefig=False):

        import matplotlib.pyplot as plt

        plt.pcolormesh(self.times, self.frequencies, self.magnitude, cmap='inferno')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time in sec')
        plt.colorbar(format='%-2.0f dB')
        plt.title(self.name)
        if savefig:
            plt.savefig("./Samples/nsynth-test/specs/64nfft/%s/%s.png" % (self.instrument, self.name))
        else:
            plt.show()


def getSpectrogram(path, filename) -> Spectrogram:
    check = re.search('(.*)(\\.db)', filename)
    if check:
        filename = check[1]
    s = shelve.open("%s/%s" % (path, filename))
    spec = s['spec']
    s.close()
    return spec


def concatenate_spectrograms(dataset, instruments=None, framed_data=False, all_data=False, limit=None, cut_silence=False, desired_pitch=None):
   
    filenames = np.array([])

    if instruments is None:
        instruments = [f for f in os.listdir("./SpectrogramData.nosync/%s" % dataset) if not f.startswith('.')]
    
    for instrument in instruments:
        instrument_path = "./SpectrogramData.nosync/%s/%s" % (dataset, instrument)
        filenames = np.append(filenames, [f for f in os.listdir(instrument_path) if not f.startswith('.')])


    if desired_pitch is not None:
        filenames = get_instruments_of_one_pitch(filenames, desired_pitch)

    np.random.shuffle(filenames)

    if limit is not None:
        filenames = filenames[:limit]


    res = []
    targets = []
    phases = np.empty((0,513))
    power_refs = []
    i = 0
    for name in filenames:

        spec = None
        for instrument in instruments:
            instrument_in_name = instrument in name
            if instrument_in_name:
                i_path = "./SpectrogramData.nosync/%s/%s" % (dataset, instrument)
                spec = getSpectrogram(i_path, name)
            
        '''
        
        if instrument_two is not None and re.match(instrument_two,name):
            spec = getSpectrogram(path_2, name)
        else:
            spec = getSpectrogram(path, name)'''
        

        if spec is None: continue

        # convert to 3d Array and concat (if data is famed)
        if framed_data:
            mag = spec.magnitude.astype(np.float32)[np.newaxis,:, :]
            targets.append(spec.name)
            phases = np.append(phases, spec.phase, axis=1)
            power_refs.append(spec.ref_power)
            res.append(mag)
        else:
            if all_data: phases = np.append(phases, spec.phase.transpose(), axis=0)
            for frame in spec.magnitude.transpose():
                if cut_silence and frame.max() == -120.0:
                    continue
                mag = frame.astype(np.float32)[np.newaxis, :]
                targets.append(spec.name)
                if all_data: power_refs.append(spec.ref_power)
                res.append(mag)
        
        i += 1
        print("Concat Process: %d/%d" % (i, len(filenames)), end='\r')


        #mag_phase = np.concatenate((mag, phase), axis=2)



    if all_data:
        return np.array(res), targets, phases, power_refs
    else:
        return np.array(res), targets


def concatenate_spectrograms_2D(dataset, instruments=None, framed_data=False, all_data=False, limit=None,
                                cut_silence=False, desired_pitch=None, overlap=False, samples=None):
    filenames = np.array([])

    if instruments is None:
        instruments = [f for f in os.listdir("./SpectrogramData.nosync/%s" % dataset) if not f.startswith('.')]

    for instrument in instruments:
        instrument_path = "./SpectrogramData.nosync/%s/%s" % (dataset, instrument)
        filenames = np.append(filenames, [f for f in os.listdir(instrument_path) if not f.startswith('.')])

    if desired_pitch is not None:
        filenames = get_instruments_of_one_pitch(filenames, desired_pitch)

    if samples is not None:

        if all(s in filenames for s in samples):
            filenames = samples
        else:
            print("Samples not available in dataset")

    np.random.shuffle(filenames)

    if limit is not None:
        filenames = filenames[:limit]

    res = []
    targets = []
    phases = []
    power_refs = []
    i = 0
    for name in filenames:

        spec = None
        for instrument in instruments:
            instrument_in_name = instrument in name
            if instrument_in_name:
                i_path = "./SpectrogramData.nosync/%s/%s" % (dataset, instrument)
                spec = getSpectrogram(i_path, name)

        if spec is None: continue

        # slice spectrogram in equal junks
        mag = spec.magnitude.transpose()
        frame_len = 3

        pad = frame_len - mag.shape[0] % frame_len

        if not overlap and pad < frame_len:
            for _ in range(0, pad):
                mag = np.vstack((mag, np.repeat(-120.0, 513)))

        f_count = mag.shape[0] // frame_len


        if not overlap:
            for idx in range(0, mag.shape[0], frame_len):
                f = mag[idx:idx + frame_len].astype(np.float32)[np.newaxis, :, :]
                if cut_silence and f.max() == -120.0:
                    continue
                if all_data: phases.append(phases[idx:idx + frame_len])
                res.append(f)
                targets.append(spec.name)
        else:
            for idx in range(0, mag.shape[0]-frame_len):
                if mag[idx].max() == -120.0: break
                f = mag[idx:idx + frame_len].astype(np.float32)[np.newaxis, :, :]
                res.append(f)
                targets.append(spec.name)



        if all_data:
            phases.append(spec.phase)
            power_refs.append(spec.ref_power)

        i += 1
        print("Concat Process: %d/%d" % (i, len(filenames)), end='\r')

        # mag_phase = np.concatenate((mag, phase), axis=2)

    if all_data:
        return np.array(res), targets, phases, power_refs
    else:
        return np.array(res), targets


def extract_power_refs(labels):

    base_path = 'SpectrogramData/%s' % 'keyboard_synthetic'
    refs = []
    for label in labels:
        spec = getSpectrogram(base_path, label)
        refs.append(spec.ref_power)

    return refs
