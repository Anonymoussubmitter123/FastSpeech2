import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )  # sort排序，drop_last最后数据不够一个batch时，舍弃掉它
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    """
    assert语句用于检查条件是否为真，如果条件为假，就会抛出AssertionError异常，程序停止执行。
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,  # shuffle是DataLoader中的一个参数，用于控制每个epoch时数据集是否被随机打乱。提高泛化能力。
        collate_fn=dataset.collate_fn,
        # 通常情况下，输入的样本长度不一，需要将它们填充到同一长度。collate_fn 函数的作用就是实现这个功能，
        # 它将一个列表中的样本进行处理，返回一个填充后的 batch。
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    '''
    model, optimizer = get_model(args, configs, device, train=True)：使用 get_model() 函数获取模型和优化器，
    并传入参数 train=True 表示这是在训练模式下初始化模型和优化器。
    model = nn.DataParallel(model)：如果有多个 GPU，使用 nn.DataParallel() 将模型并行在多个 GPU 上运行。
    num_param = get_param_num(model)：使用 get_param_num() 函数获取模型的参数数量。
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)：使用 FastSpeech2Loss 类初始化损失函数，并将其移到设备上。
    print("Number of FastSpeech2 Parameters:", num_param)：输出模型的参数数量。
    '''

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    """
    for p in train_config["path"].values():：遍历 train_config["path"] 字典中的所有值；
    os.makedirs(p, exist_ok=True)：如果文件夹不存在则创建文件夹；
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")：设置训练日志保存路径；
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")：设置验证日志保存路径；
    os.makedirs(train_log_path, exist_ok=True)：如果训练日志文件夹不存在则创建文件夹；
    os.makedirs(val_log_path, exist_ok=True)：如果验证日志文件夹不存在则创建文件夹；
    train_logger = SummaryWriter(train_log_path)：创建训练日志记录器；
    val_logger = SummaryWriter(val_log_path)：创建验证日志记录器。

    日志记录器是用于记录训练和验证过程中的相关信息，如损失值、准确率等指标，便于分析和可视化。
    SummaryWriter 是 PyTorch 中的一个内置类，用于写入 TensorBoard 日志文件。
    """

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    """
    args.restore_step: 恢复模型的步数，即训练从哪个步数开始恢复
    grad_acc_step: 梯度累积的步数，即多少个batch的梯度累积为一次更新
    grad_clip_thresh: 梯度裁剪的阈值，用于控制梯度的范数，防止梯度爆炸
    total_step: 总共的训练步数
    log_step: 训练日志输出的步数间隔，即每训练多少步就输出一次训练日志
    save_step: 模型保存的步数间隔，即每训练多少步就保存一次模型
    synth_step: 合成音频的步数间隔，即每训练多少步就合成一次音频
    val_step: 验证模型的步数间隔，即每训练多少步就验证一次模型的性能
    """

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    """
    使用tqdm库创建了一个进度条，用于展示训练的进度，其中total表示总的训练步数，desc表示描述，position表示进度条的位置。
    接着，将进度条的当前位置设置为开始训练的步数，并更新进度条。
    """
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
