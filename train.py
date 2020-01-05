import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

PAD = '[PAD]'
pad_id = 0
logger = None


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--train_mmi', action='store_true', help="若指定该参数，则训练DialoGPT的MMI模型")
    parser.add_argument('--train_mmi_tokenized_path', default='data/train_mmi_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
    parser.add_argument('--mmi_model_output_path', default='mmi_model', type=str, required=False, help='MMI模型保存路径')
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(args, vocab_size):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    if args.pretrained_model:  # 如果指定了预训练的GPT2模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")


def preprocess_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(args.train_raw_path,
                                                                                    args.train_tokenized_path))
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
    with open(args.train_tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            # 最后一条记录不添加换行符
            if dialogue_index < len(train_data) - 1:
                f.write("\n")
    logger.info("finish preprocessing raw data,the result is stored in {}".format(args.train_tokenized_path))


def preprocess_mmi_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料的每段对话进行翻转，然后转换为用于train MMI模型的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance N[SEP]utterance N-1[SEP]utterance N-2[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing MMI raw data,raw data path:{}, token output path:{}".format(args.train_raw_path,
                                                                                        args.train_mmi_tokenized_path))
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
    with open(args.train_mmi_tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in reversed(utterances):  # 将一段对话进行翻转
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            # 最后一条记录不添加换行符
            if dialogue_index < len(train_data) - 1:
                f.write("\n")
    logger.info("finish preprocessing raw data,the result is stored in {}".format(args.train_tokenized_path))


def calculate_loss_and_accuracy(outputs, labels, device):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, device, train_list, multi_gpu, args):
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # 记录 out of memory的次数
    oom_time = 0
    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
            # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids = input_ids.to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = model.forward(input_ids=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

                if multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    # 更新参数
                    optimizer.step()
                    # 清空梯度信息
                    optimizer.zero_grad()
                    # 进行warm up
                    scheduler.step()
                    overall_step += 1
                    # 更新日志与tnesorboardX信息
                    if (overall_step + 1) % args.log_step == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, epoch + 1, loss,
                                                                                accuracy))
                        tb_writer.add_scalar('loss', loss.item(), overall_step)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if args.train_mmi:  # 当前训练MMI模型
            model_path = join(args.mmi_model_output_path, 'model_epoch{}'.format(epoch + 1))
        else:  # 当前训练对话模型
            model_path = join(args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('training finished')


def evaluate(model, device, test_list, multi_gpu, args):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
        logger.info("finishing evaluating")


def main():
    args = setup_train_args()
    # 日志同时输出到文件和console
    global logger
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)

    # 创建对话模型的输出目录
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)
    # 创建MMI模型的输出目录
    if not os.path.exists(args.mmi_model_output_path):
        os.mkdir(args.mmi_model_output_path)
    # 加载GPT2模型
    model, n_ctx = create_model(args, vocab_size)
    model.to(device)
    # 对原始数据进行预处理,将原始语料转换成对应的token_id
    if args.raw and args.train_mmi:  # 如果当前是要训练MMI模型
        preprocess_mmi_raw_data(args, tokenizer, n_ctx)
    elif args.raw and not args.train_mmi:  # 如果当前是要训练对话生成模型
        preprocess_raw_data(args, tokenizer, n_ctx)
    # 是否使用多块GPU进行并行运算
    multi_gpu = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use GPUs to train")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载数据
    logger.info("loading traing data")
    if args.train_mmi:  # 如果是训练MMI模型
        with open(args.train_mmi_tokenized_path, "r", encoding="utf8") as f:
            data = f.read()
    else:  # 如果是训练对话生成模型
        with open(args.train_tokenized_path, "r", encoding="utf8") as f:
            data = f.read()
    data_list = data.split("\n")
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)
    # 开始训练
    train(model, device, train_list, multi_gpu, args)
    # 测试模型
    evaluate(model, device, test_list, multi_gpu, args)


if __name__ == '__main__':
    main()
