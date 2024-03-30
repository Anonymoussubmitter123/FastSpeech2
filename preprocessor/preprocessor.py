import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)  # 创建目录
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        """
        提取输入音频文件的Mel-Spectrogram和能量信息。
        它首先通过调用get_mel_from_wav()函数获取音频文件的Mel-Spectrogram和能量信息。
        get_mel_from_wav()函数将输入音频信号进行短时傅里叶变换(STFT)并应用Mel滤波器组，
        生成Mel-Spectrogram。

        然后，将从音频文件中提取出的Mel-Spectrogram和能量信息的长度截断到和音频的实际长度相同。
        在这个过程中，由于get_mel_from_wav()函数返回的Mel-Spectrogram和能量信息是按照整数帧的形式组织的，
        而音频的实际长度不一定正好是整数帧的长度，因此需要进行截断。
    
        最后，将处理好的Mel-Spectrogram和能量信息返回。
        """

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            """
            具体来说，interp1d 接受两个参数： x 和 y。其中， x 是一个一维数组，表示数据点的横坐标，
            y 是一个一维数组，表示数据点的纵坐标。
            函数会返回一个函数对象，该函数对象可以对新的横坐标进行插值，得到对应的纵坐标。
            插值方法默认为线性插值，也可以通过 kind 参数设置其他插值方法。
            """
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):  # enumerate()获得列表数组的指针和值
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])  # mean()用于计算数组或矩阵的平均值。
                    """
                    这行代码的意思是将pos到pos+d之间的pitch值求平均，然后将这个平均值赋给pitch[i]，
                    其中i是当前循环迭代的索引值。pos是一个起始索引值，用于确定每个音素在pitch中对应的位置。
                    在for循环中，d是当前音素的持续时间，即持续几个时间步长。
                    
                    这行代码的目的是将每个音素对应的pitch值替换为在该音素持续时间内的平均值，
                    以平滑pitch轮廓。
                    """
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]
            """
            这部分代码是为了将音高（pitch）和能量（energy）信息对齐到每个音素的时间段上，
            便于后续使用。具体而言，首先通过duration数组中记录的每个音素的时长信息，
            将pitch和energy信息平均到每个音素的时长上，得到每个音素的pitch和energy信息，
            然后再根据duration数组的长度截取每个音素的pitch和energy信息。
            
            这么做的目的是将原本按照时间点采样的pitch和energy信息对齐到每个音素上，
            以便后续的处理和分析。
            """

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
        """
        这段代码是将处理后的语音信息保存为npy文件，其中每个npy文件保存一种信息
        （包括持续时间duration、语音的音调pitch、语音能量energy以及mel频谱）。
        
        这些文件的命名格式为"speaker-basename.npy"，其中speaker表示说话人的编号，
        basename表示语音文件的名称。
        
        函数返回一个元组，包含4个信息，分别是语音的基本信息、去除异常值后的音调、
        去除异常值后的能量和mel频谱的长度。
        """
        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            """
            durations 列表记录每个音素在语音信号中所占的时间步数。
            这是通过计算音素在语音信号中的起始和结束位置，然后将其转换为时间步数来实现的。
            
            具体来说，
            这个计算过程包括以下几个步骤：计算音素在语音信号中的起始和结束采样点数，即 s 和 e。
            将起始和结束采样点数转换为起始和结束时间步数，即 s * self.sampling_rate / self.hop_length
            和 e * self.sampling_rate / self.hop_length。
            将结束时间步数减去起始时间步数，并将结果四舍五入为整数，得到该音素在语音信号中所占的时间步数。
            将该时间步数添加到 durations 列表中。
            """

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        """
        这个方法的目的是通过去除异常值来减少对模型训练的影响。
        在一些数据中，存在着一些与其他值相差较大的异常值，如果这些异常值被保留下来，
        它们可能会对训练模型产生很大的负面影响，导致模型波动性大，精度不稳定。
        因此，通过去除这些异常值，可以让训练数据更加平滑，更好地适应模型。
        假设我们有以下一组数据：
        [2, 3, 5, 7, 9, 11, 13, 15, 17, 20]
        我们想要去除其中的离群值（outliers），可以使用IQR方法来进行去除。
        首先，我们需要计算出这组数据的四分位数（quartiles）：
            第一四分位数（Q1）：该数据中位于中位数左侧的数的中位数（即第25个百分位数）。
            第二四分位数（Q2）：该数据中的中位数（即第50个百分位数）。
            第三四分位数（Q3）：该数据中位于中位数右侧的数的中位数（即第75个百分位数）。
        对于上述数据，计算得到：
            Q1 = 4
            Q2 = 10
            Q3 = 16
        接下来，我们可以计算出该数据的IQR值，即IQR = Q3 - Q1 = 12。
        根据IQR值，我们可以定义一个上限和下限，将不在这个范围内的数据视为离群值：
            下限 = Q1 - 1.5 * IQR = -8
            上限 = Q3 + 1.5 * IQR = 28
        因此，在该数据中，2和20被认为是离群值，应该被去除。使用IQR方法去除离群值后，最终得到的数据为：
        [3, 5, 7, 9, 11, 13, 15, 17]
        在上面的代码中，remove_outlier函数就是用来实现这个功能的，其中的values参数就是要进行处理的一组数据。
        """
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
    """
    这个 normalize 函数的主要作用是对输入的语音信号进行归一化处理，使其具有零均值和单位方差。
    具体实现方式为：

    遍历 in_dir 目录下的所有语音信号文件。
    对于每个语音信号文件，读入数据，减去 mean 后再除以 std，使得数据具有零均值和单位方差。
    保存处理后的数据到原文件中。
    计算处理后的数据中的最大值和最小值，并返回。

    这么做的目的是为了使得模型训练更加稳定，因为归一化后的数据具有统一的分布范围和尺度，
    可以降低不同特征之间的比例差异，使得优化器更容易找到最优解。
    同时，由于深度学习模型对于输入数据的尺度和范围比较敏感，
    因此归一化操作也可以防止梯度消失或梯度爆炸等训练过程中的数值问题。
    """
