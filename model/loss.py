import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """
    """
    接收两个输入参数 inputs 和 predictions，并返回一个元组，包含了总损失和各项损失的数值。

    该损失函数主要包含了五项损失，即 mel_loss、postnet_mel_loss、duration_loss、pitch_loss 和 energy_loss。
    
    其中，mel_loss 和 postnet_mel_loss 是采用 L1 损失函数计算预测语音频谱和目标语音频谱之间的差距；
    而duration_loss、pitch_loss 和 energy_loss 分别采用均方误差损失函数计算预测的语音时长、语音音调和语音能量与目标值之间的差距。
    
    在 forward 方法中，首先通过输入参数 inputs 和 predictions 获取各项目标值和预测值，并对其中的一些变量进行处理，
    
    例如使用 log 函数对语音时长进行变换。然后对于每一项损失，根据不同的特征类型（音素级或帧级）选择不同的目标值和预测值，
    并采用相应的损失函数计算差距。最后将各项损失相加得到总损失，并将各项损失的数值返回。
    
    
    L1损失和均方误差损失是深度学习中两种广泛使用的损失函数，它们的主要不同点在于它们如何惩罚预测值和目标值之间的差异。
    
    L1损失，也称为绝对误差损失，计算预测值和目标值之间的绝对差异，即预测值和目标值之间的距离的绝对值之和。它的数学公式为：
    L1 loss = ∑|y - ŷ|
    其中y是目标值，ŷ是模型的预测值。L1损失对异常值（即预测值和目标值之间的差异很大的点）比均方误差损失更鲁棒，
    因为它不会对差异的平方进行惩罚，而是对其进行线性惩罚。

    均方误差损失，也称为平方误差损失，计算预测值和目标值之间的平方差异，即预测值和目标值之间的距离的平方和。它的数学公式为：
    MSE loss = ∑(y - ŷ)^2
    均方误差损失对于大多数情况下都很有效，特别是对于连续值的回归问题。但是，它对异常值更加敏感，
    因为它对差异的平方进行惩罚，导致异常值的影响更大。
    总之，L1损失和均方误差损失的选择取决于具体的问题和数据集的特点。如果数据集中包含异常值，则L1损失通常是更好的选择。
    如果数据集中不存在异常值，并且预测值和目标值之间存在线性关系，则均方误差损失可能是更好的选择。
    
    """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        """
         这两句代码主要是为了裁剪 mel_targets 和 mel_masks 的最后一个维度，使得它们的长度和 mel_predictions 和
         postnet_mel_predictions 的长度一致。具体来说，mel_targets 是目标的 Mel 频谱，mel_masks 是掩码矩阵，
         用于标记 Mel 频谱中无效的部分。由于 Mel 频谱是由声学特征（如语音帧）通过一些信号处理操作生成的，
         因此它们的长度通常是不同的。但在这个模型中，我们需要将目标 Mel 频谱与模型的预测 Mel 频谱对齐，
         因此需要将目标 Mel 频谱和掩码的长度裁剪为与预测 Mel 频谱长度一致。这里 mel_masks 的第二个维度被裁剪到
         与 mel_predictions 和 postnet_mel_predictions 的长度一致，然后这个掩码矩阵被用来对预测的 Mel 频谱进行掩码操作。
        """

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        """
        在深度学习中，通常只需要对模型中的参数求导，而对于目标张量等不需要训练的张量，设置 requires_grad=False 可以避免无意义
        的计算和浪费计算资源。
        """

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
