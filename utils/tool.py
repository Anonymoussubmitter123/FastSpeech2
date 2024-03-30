import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_device(data, device):
    if len(data) == 13:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            avg_mel_phs,
            # lang_ids,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        avg_mel_phs = torch.from_numpy(avg_mel_phs).float().to(device)
        # lang_ids = torch.from_numpy(lang_ids).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            avg_mel_phs,
            # lang_ids
        )

    if len(data) == 7:
    # if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, mel) = data
        # (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mel = torch.from_numpy(mel).float().to(device)
        mel = torch.transpose(mel, 1, 2)
        # languages = torch.from_numpy(lang_ids).long().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, mel)
        # return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/phone_level_loss", losses[6], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)
    """例如，假设有一个长度为3的音频特征值数组[1, 2, 3]，以及一个持续时间数组[2.5, 3.5, 1.5]。函数将根据持续时间将1重复2次、
    2重复3次、3重复1次，然后将所有的值组成一个新的数组[1, 1, 2, 2, 2, 3]并返回。这样，新的数组中的每个值都对应于原始音频中的一个phoneme。"""


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    """此处的predictions的矩阵行坐标和Adaspeech模型的输出对应。"""
    basename = targets[0][0]
    src_len = predictions[9][0].item()
    mel_len = predictions[10][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

        # plt.imshow(mel_prediction.cpu().detach().numpy()[0, :, :], interpolation='none', cmap=plt.cm.jet, origin='lower')
        from skimage.transform import resize  # 导入 resize 方法

        # 假设 mel_prediction 是大小为 (1, H, W) 的 PyTorch 张量
        # 将其转换为 Numpy 数组，并缩小高度到 36 像素
        # mel_pred_np = mel_prediction.cpu().detach().numpy()
        # H, W = mel_pred_np.shape[1:]
        # mel_prediction = resize(mel_pred_np[0], output_shape=(36, int(36 * W / H)))

        plt.imshow(mel_prediction.cpu().detach().numpy(), interpolation='none',  origin='lower')
        plt.xticks(np.arange(0, mel_prediction.shape[1] + 1, 100))
        plt.yticks(np.arange(0, mel_prediction.shape[0] + 1, 50))
        plt.savefig(os.path.join(path, "mel_{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    # 创建一个包含多个子图的figure对象。子图的数量等于data列表的长度，每个子图都是一个包含一个Axes对象的numpy数组。
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    # 将传递的音高和能量统计信息元组解包为各个变量。
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean
    # 将统计信息中的标准化音高最小值和最大值转换为实际音高值。

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]  # 将当前数据元素解包为mel、pitch和energy三个变量。
        pitch = pitch * pitch_std + pitch_mean  # 将当前音高曲线中的标准化值转换为实际音高值。
        axes[i][0].imshow(mel, origin="lower")  # 将当前mel spectrogram绘制在当前子图中。
        axes[i][0].set_aspect(2.5, adjustable="box")  # 设置当前子图的纵横比为2.5，以便适当地显示mel spectrogram的形状。
        axes[i][0].set_ylim(0, mel.shape[0])  # 将当前子图的y轴范围设置为mel spectrogram的高度。
        axes[i][0].set_title(titles[i], fontsize="medium")  # 将当前子图的标题设置为titles列表中相应位置的值，使用“medium”字体大小。
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)  # 将当前子图的刻度线设置为小字体大小，并将左侧和左侧标签
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
