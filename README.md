# GPT2 for Chinese chitchat

## 项目描述
- 本项目使用GPT2模型对中文闲聊语料进行训练，使用 HuggingFace的[transformers](https://github.com/huggingface/transformers)实现GPT2模型的编写与训练。
- 在闲暇时间用 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)模型训练了几个长文本的生成模型，并且精读了一遍作者的源码，获益匪浅，加深了自己对GPT2生成模型的一些理解，于是将GPT2模型用于闲聊对话的生成，非常感谢作者的分享。
- 本项目中沿用了原项目中的部分结构和一些命名方式，同时也对很多代码细节做出了自己实现。
- 解码器的逻辑使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- 代码中给出了许多详细的中文注释，方便大家更好地理解代码(能力有限，可能有些代码或注释有误，望大家不吝赐教)

## 运行环境
python3.6、 transformers==2.1.1、pytorch==1.3.1

## 项目结构
- config:存放GPT2模型的参数的配置文件
- data
    - train.txt:默认的原始训练集文件，存放闲聊语料 
    - train_tokenized.txt:对原始训练语料进行tokenize之后的文件
- model:存放模型
- sample:存放人机闲聊生成的历史聊天记录
- vocabulary:存放GPT2模型的字典
- train.py:训练代码
- interact.py:人机交互代码


## 模型参数(详见config/model_config_dialogue_small.json文件)
- initializer_range: 0.02
- layer_norm_epsilon: 1e-05
- n_ctx: 300
- n_embd: 768
- n_head: 12
- n_layer: 10
- n_positions: 300
- vocab_size: 13317


## 使用方法
在项目根目录下创建data文件夹，将原始训练语料命名为train.txt，存放在该目录下。train.txt的格式如下，每段闲聊之间间隔一行。

![avatar](image/chitchat_demo.png)

训练模型：
``` bash
python train.py --epochs 30 --batch_size 8 --device 0,1 --raw(若要对原始训练语料进行tokenize，则要指定--raw参数。若要用GPU训练，则通过--device指定GPU)
或
python train.py --epochs 30 --batch_size 8 --no_cuda --raw(指定--no_cuda参数，则使用CPU训练，速度要慢得多)
或
python train.py --epochs 30 --batch_size 8 --no_cuda(若已经对原始语料进行tokenize，可以不用指定--raw，避免重复tokenize，节约时间)
```
人机交互：
``` bash
python interact.py --no_cuda --model_path path_to_your_model --max_history_len 3(由于闲聊对话生成的内容长度不是很长，因此生成部分在CPU上跑速度也挺快的)
```
更多的参数介绍，可直接看train.py和interact.py中的setup_train_args()函数中的参数说明

## 生成样例
#### Sample 1:

|Role | Response |
|---------|--------|
|user | 你好啊 |
| chatbot | 你好，请问您是？|
|user | 我是你的主人啊 |
| chatbot | 你是狗吗？ |
|user |你才是狗 |
| chatbot | 你不是狗我是你的小主人 |
|user | 你这个大笨蛋 |
| chatbot | 你大笨蛋，你是想挨打了 |
|user | 你全家都是大笨蛋 |
| chatbot | 我是你爸爸 |
|user | 算了，不跟你吵了，你在干嘛呢|
| chatbot | 我在上班啊，你还没下班呢啊你 |
|user | 我周末要做作业啊 |
| chatbot | 不是才交了吗，早晚做完啊|
|user | 好多作业啊，还没做完呢 |
| chatbot | 我在抄，我妈让我周五交，还有事 |
|user |你就不怕被你妈妈骂吗 |
| chatbot | 不怕 |
|user | 你周末有去哪里玩吗|
| chatbot | 没时间啊，忙得很啊|
|user | 忙着干嘛呢 |
| chatbot | 不知道啊，我周末没有作业，我想睡觉，然后还得找人 |


#### Sample 2:

|Role | Response |
|---------|--------|
|user | 你好啊 |
| chatbot | 你好我好大家好|
|user | 你在干嘛呢 |
| chatbot | 你不上课啊 |
|user |今天是周末啊，笨蛋 |
| chatbot | 我在学校呢 |
|user | 我也是，周末还要做作业 |
| chatbot | 我还没写呢我们都还没写|
|user | 周末你有去哪里玩吗 |
| chatbot | 没有去南京 |
|user |南京有什么好玩的啊 |
| chatbot |  我们没去南京|
|user | 你刚刚又说你去了南京，你是不是耍我啊|
| chatbot | 我刚刚才到南京|
|user | 你好笨啊 |
| chatbot | 我没有 |
|user |你是大笨蛋 |
| chatbot | 你才是大笨蛋 |
|user | 你全家都是大笨蛋 |
| chatbot | 你个傻冒|
|user | 你个智障 |
| chatbot | 哈哈哈 |

#### Sample 3:

|Role | Response |
|---------|--------|
|user | 在干嘛 |
| chatbot | 睡觉|
|user |这么早就睡觉啦 |
| chatbot | 嗯啊你也早点睡啊 |
|user |我再玩一会儿手机 |
| chatbot | 早点休息晚安 |
|user | 拜拜啦 |
| chatbot | 晚安晚安|
|user |多喝热水 |
| chatbot | 哈哈哈 |
|user | 多喝热水 |
| chatbot | 好的咯|

#### Sample 4:

|Role | Response |
|---------|--------|
|user | 在吗 |
| chatbot | 在！|
|user |我不开心 |
| chatbot | 怎么啦 |
|user |最近好累啊 |
| chatbot | 是挺累的 |
|user | 作业好多啊 |
| chatbot | 还好还好|
|user |我觉得我快要挂科了 |
| chatbot | 我也是啊，但是我觉得我快熬出头了 |
|user | 不说了，我还有好多作业没写 |
| chatbot | 我也是啊好多啊|
|user | 那先这样，下次再聊，再见了 |
| chatbot | 好哒|

#### Sample 5:

|Role | Response |
|---------|--------|
|user | 你去过广州吗 |
| chatbot | 去过呀，但是没有找到工作呢|
|user |那你去过北京吗 |
| chatbot | 没有啊 |
|user |北京有什么好玩的 |
| chatbot | 没有你想象的那么好 |
|user | 是吗，我还没过北京 |
| chatbot | 那里我也不怎么熟，我去过北京|
|user |北京天安门呢 |
| chatbot | 没去过，不过可以去玩玩 |

## 闲聊语料
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料训练模型 | [百度网盘【提取码:jk8d】](https://pan.baidu.com/s/1mkP59GyF9CZ8_r1F864GEQ) |由作者[GaoQ1](https://github.com/GaoQ1)提供的比较高质量的闲聊数据集，整理出了50w个多轮对话的语料|

## 模型分享
|模型 | 百度网盘 |提取码|模型描述|
|---------|--------|--------|--------|
|50w中文闲聊语料训练模型 | [百度网盘](https://pan.baidu.com/s/1EZMF0QcxXBeWF8HMoNpyfQ) |gi5i|闲聊语料为67M，包含50w个多轮对话，用两块1080Ti,大概跑了五六天(应该没有记错)，训练了40个epoch，最终loss在2.0左右，继续训练的话，loss应该还能继续下降。|


模型使用方法：把下载好的模型文件pytorch_model.bin和config.json放在model目录下(否则需要通过--model_path参数指定模型的路径)，执行如下命令:
``` bash
python interact.py --no_cuda --model_path path_to_your_model --max_history_len 5(由于闲聊对话生成的内容长度不是很长，因此生成部分在CPU上跑速度也挺快的。根据需求调整max_history_len参数)
``` 
输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的sample.txt文件中

## interact.py的参数
执行interact.py时，可以尝试通过指定topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果

## 不足之处
虽然使用了多轮历史对话作为response的生成条件，但是chatbot还是不能很好地掌握上下文的逻辑。比如Sample3与Sample5中，chatbot无法知道自己"是否在南京","是否去过北京"。对于下面Sample6和Sample7这种重复性的提问(user重复输入一样的对话内容)，也无法给出一致的答案。
#### Sample 6:
|Role | Response |
|---------|--------|
|user | 你在干嘛 |
| chatbot |在睡觉 |
|user |你在干嘛 |
| chatbot | 上班 |
|user |你在干嘛 |
| chatbot | 想我了？ |
|user | 你在干嘛 |
| chatbot | 我在看歌手|
|user |你在干嘛 |
| chatbot | 在家里过节呢 |
|user | 你在干嘛 |
| chatbot | 看球赛|
|user | 你在干嘛 |
| chatbot | 我在想你|
|user | 你在干嘛 |
| chatbot | 在吃饭|
|user | 你在干嘛 |
| chatbot | 搞学术|
|user | 你在干嘛 |
| chatbot | 我在看电视|

#### Sample 7:
|Role | Response |
|---------|--------|
|user | 天气怎样 |
| chatbot |不错 |
|user |天气怎样 |
| chatbot | 挺好的 |
|user |天气怎样 |
| chatbot | 不错不错，挺美的 |
|user | 天气怎样 |
| chatbot | 我在看歌手|
|user |天气怎样 |
| chatbot | 不错不错 |
|user | 天气怎样 |
| chatbot | 下大雨，好冷啊|
|user | 天气怎样 |
| chatbot | 风和日丽的不要不要的|
|user | 天气怎样|
| chatbot | 下大雨|

chatbot偶尔也会"智商离线"，生成的内容"惨不忍睹"，如Sample8
#### Sample 8:
|Role | Response |
|---------|--------|
|user | 在干嘛呢 |
| chatbot |上班呢 |
|user |这么晚还上班啊 |
| chatbot | 是滴，你呢 |
|user |我再上课啊 |
| chatbot | 那么辛苦，吃饭不 |
|user | 你要请我吗 |
|user | 你来 |
| chatbot |我没钱啊 |
|user |那是什么狗屁 |
| chatbot | 你好笨啊 |
|user |我们都是要保养保养，保养，保养，保养，保养，牙套牙 |
| chatbot | 你好笨啊 |
|user | 好好说话 |
| chatbot | 拜拜 |
|user | 爱你呦芸芸 |

## Future Work
更多地在解码器上下功夫，比如使用互信息(mutual information)：训练一个额外的网络，给定一个reponse，该网络能够计算出概率P(Source|response)，Source为response的上文。该网络的目的就是对于生成的多个response，选出P(Source|response)最大的response作为最终的回复。


## Reference
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)




