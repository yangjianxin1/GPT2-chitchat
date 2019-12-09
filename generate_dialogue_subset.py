import argparse
from os.path import join
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def generate_subset():
    """
    用于生成训练子集
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--subset_size', default=500000, type=int, required=False, help='要获取的对话数据子集的规模')
    parser.add_argument('--subset_data_path', default='data', type=str, required=False,
                        help='数据子集文件路径,指定文件的父目录')
    args = parser.parse_args()
    with open(args.raw_data_path, "r", encoding="utf8") as f:
        data = f.read()
    dialogues = data.split("\n\n")
    subset_size = min(len(dialogues), args.subset_size)

    with open(join(args.subset_data_path, "train_{}w.txt".format(int(subset_size / 10000))), "w", encoding="utf8") as f:
        print("generating subset,please wait a few seconds ")
        for dialogue_index, dialogue in enumerate(dialogues):
            if dialogue_index >= subset_size:
                break
            for utterance in dialogue.split("\n"):
                f.writelines(utterance + "\n")
            f.writelines("\n")


def compute_dialogue_length():
    """
    查看聊天语料中的dialogue的长度分布
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    args = parser.parse_args()
    with open(args.raw_data_path, "r", encoding="utf8") as f:
        data = f.read()
    dialogues = data.split("\n\n")
    # 统计各个dialogue的长度
    dialogues_lengths = [len(dialogue.replace("\n", "")) for dialogue in dialogues]
    counter = Counter(dialogues_lengths)  # {label:sum(label)}
    dialogue_length_arr = list(counter)
    num_arr = [counter[element] for element in list(counter)]
    print(counter[300])

    x_major_locator = MultipleLocator(100)  # MultipleLocator用于设置刻度间隔
    # y_major_locator = MultipleLocator(20000)
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为10的倍数
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('dialogue length')
    plt.ylabel('number of dialogue')
    # plt.plot(dialogue_length_arr, num_arr, c='green')
    plt.scatter(dialogue_length_arr, num_arr)
    plt.show()


if __name__ == '__main__':
    compute_dialogue_length()
