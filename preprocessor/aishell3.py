import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_align(config):  # prepare_align是一个预处理函数，主要是用于数据准备和对齐。
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for dataset in ["train", "test"]:
        print("Processing {}ing set...".format(dataset))
        with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
            for line in tqdm(f):
                wav_name, text = line.strip("\n").split("\t")
                speaker = wav_name[:7]
                text = text.split(" ")[1::2]
                """
                将文本分割成单词列表，然后从中提取所有奇数位置的单词，即将标签提取出来。
                
                例如，假设文本是 "SIL I AM READY FOR THE EXERCISE"，则分割后得到 
                ['SIL', 'I', 'AM', 'READY', 'FOR', 'THE', 'EXERCISE']，
                然后通过 [1::2] 提取奇数位置的单词，即 ['I', 'READY', 'THE']，这些就是标签。
                """
                wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, wav_name),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),"w",) as f1:
                        f1.write(" ".join(text))
                    """
                    对指定目录下的数据集（包括训练集和测试集）进行预处理，将音频文件转换成标准的wav格式，
                    进行重采样，保存到指定输出目录，并将音频的对应文本保存到与音频文件同名的lab文件中。
                    
                    具体来说：
                    
                    读取数据集中的content.txt文件，该文件记录了每个音频文件名以及其对应的文本信息。
                    根据音频文件名获取所在的路径，读取音频文件，并将其重采样为指定采样率。
                    同时将采样后的音频值归一化并缩放至指定的最大值（max_wav_value）。
                    将处理后的音频文件保存到输出目录，并将音频的对应文本保存到与音频文件同名的lab文件中，
                    每个单词之间用空格隔开。
                    """