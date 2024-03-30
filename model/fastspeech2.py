import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        """其中，get_mask_from_lengths函数的作用是从序列长度中生成一个掩码(mask)，用于在神经网络中对序列进行填充(padding)。
            这个掩码能够过滤掉填充部分的信息，使得神经网络只关注真实的输入。"""

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        """ 掩码(mask)在神经网络中通常用于处理变长序列数据。当我们处理文本、语音、视频等序列数据时，不同的输入序列可能具有不同的长度。
            为了方便处理这些序列，我们通常需要将它们进行填充(padding)使它们的长度一致。但是，这样做会在序列中添加许多无用的0，
            这会使神经网络学习到错误的信息，从而影响模型的性能。为了解决这个问题，我们需要一种方法来避免网络学习填充的无用信息。
            这时，掩码就派上用场了。掩码是一个与输入序列维度相同的二元矩阵，其中0表示该位置是填充，1表示该位置是真实的数据。
            在神经网络中，我们通常使用掩码来过滤掉填充部分的信息，只保留真实的数据。这样，网络就只会学习到真实的数据，
            而不会被填充的无用信息所干扰。
        
            在实际应用中，我们通常会将掩码与输入数据一起输入到神经网络中进行训练，以提高模型的性能。"""

        output = self.encoder(texts, src_masks)
        """这段代码中，一个神经网络模型的encoder被调用，它有两个输入：texts和src_masks。texts是一个包含文本序列的张量，
        src_masks是上一个问题中提到的掩码，用于对文本序列进行填充。在模型中，encoder将输入的文本序列通过一系列的神经网络层进行处理，
        最终输出一个表示文本序列的向量output。这个向量可以被用于后续的任务，比如文本分类或者序列生成等。"""

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        """ 这段代码是一个条件语句，它检查 self.speaker_emb 属性是否为 None。如果不是 None，则代码将对 output 张量进行操作。
            进行的操作涉及使用 speakers 张量作为输入调用 self.speaker_emb() 方法。
            然后，使用两个操作修改生成的张量：unsqueeze(1) 和 expand()。unsqueeze(1) 方法在索引1处向张量添加一个新维度，
            这实际上将形状为 (batch_size, num_features) 的二维张量转换为形状为 (batch_size, 1, num_features) 的三维张量。
            expand(-1, max_src_len, -1) 方法沿着张量的维度进行扩展。在这种情况下，它将沿着第二个维度(1)复制张量，
            以获得形状为 (batch_size, max_src_len, num_features)。
            
            最后，原始的 output 张量与从 self.speaker_emb() 方法获得的修改后的张量进行逐元素相加。"""
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        """
        Args:
            encoder_output: encoder输出的张量
            src_mask: 源序列掩码张量
            mel_mask: 目标语音序列掩码张量
            max_mel_len: 目标语音序列最大长度
            p_target: 目标语音序列的基频（pitch）特征，用于指导模型生成输出序列的基频
            e_target: 目标语音序列的能量（energy）特征，用于指导模型生成输出序列的能量
            d_target: 目标语音序列的持续时间（duration）特征，用于指导模型生成输出序列的持续时间
            p_control: 基频控制参数，用于调整生成序列的基频特征
            e_control: 能量控制参数，用于调整生成序列的能量特征
            d_control: 持续时间控制参数，用于调整生成序列的持续时间特征
        Returns:
            output: 经过持续时间调整后的encoder输出张量
            p_predictions: 基频预测值张量
            e_predictions: 能量预测值张量
            log_d_predictions: 对数化的持续时间预测值张量
            d_rounded: 取整后的持续时间预测值张量
            mel_lens: 目标语音序列长度张量
            mel_mask: 经过持续时间调整后的目标语音序列掩码张量
        """
        """
        VarianceAdaptor主要实现了三个方面的功能：pitch contour变换、energy contour变换以及duration变换。
        接下来我将分别解释一下各个步骤的实现细节：

    Pitch contour变换

该部分实现主要包括了三个步骤：

（1）使用一个逐帧的pitch predictor来对输入文本的音高进行建模，得到一个预测的pitch contour。

（2）使用一个控制函数来对预测的pitch contour进行变换。这里的控制函数是一个折线函数，其中每个拐点都代表了一个离散的pitch scale factor。
通过控制函数可以实现对pitch contour的缩放、偏移等操作。

（3）使用一个后处理模块，对变换后的pitch contour进行平滑处理。

具体实现细节可以参考代码中variance_adaptor.py文件中的pitch_transform()方法。

    Energy contour变换

该部分实现主要包括了两个步骤：

（1）使用一个逐帧的energy predictor来对输入文本的能量进行建模，得到一个预测的energy contour。

（2）使用一个控制函数来对预测的energy contour进行变换。这里的控制函数和pitch contour变换的控制函数是相同的。
具体实现细节可以参考代码中variance_adaptor.py文件中的energy_transform()方法。

    Duration变换

该部分实现主要包括了两个步骤：

（1）使用一个逐帧的duration predictor来对输入文本的持续时间进行建模，得到一个预测的duration contour。

（2）使用一个控制函数来对预测的duration contour进行变换。这里的控制函数和pitch contour变换的控制函数是相同的。
具体实现细节可以参考代码中variance_adaptor.py文件中的duration_transform()方法。

除了上述三个主要的变换操作之外，VarianceAdaptor还实现了一些辅助功能，比如mask生成、输入特征的拼接等。
具体实现细节可以参考代码中variance_adaptor.py文件中的forward()方法。
        """
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output
        """最后，将 output 作为输入传递给了 self.postnet() 方法，并对返回结果和 output 进行了逐元素相加。
        这个操作的目的是为了增强输出声音的清晰度，通常被称为声学后处理（post-processing）。"""

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )