# GPT2 for Chinese chitchat

## UPDATE 2020.01.09
添加50w闲聊语料与预训练模型的GoogleDrive的下载地址

## UPDATE 2019.12.17
基于微软的论文[DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)添加了MMI Model(maximum mutual information scoring function),对dialogue model生成的多个response进行筛选


## 项目描述
- 本项目使用GPT2模型对中文闲聊语料进行训练，使用 HuggingFace的[transformers](https://github.com/huggingface/transformers)实现GPT2模型的编写与训练。
- 在闲暇时间用 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)模型训练了几个长文本的生成模型，并且精读了一遍作者的源码，获益匪浅，加深了自己对GPT2生成模型的一些理解，于是将GPT2模型用于闲聊对话的生成，非常感谢作者的分享。
- 本项目中沿用了原项目中的部分结构和一些命名方式，同时也对很多代码细节做出了自己实现。
- 解码器的逻辑使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- 根据微软的DialoGPT的思想，在项目中添加了互信息。训练了两个模型:Dialogue Model与MMI Model(maximum mutual information scoring function)。首先使用Dialogue Model生成多个候选response，然后使用MMI Model从候选response中，选取loss最小的作为最终的response
- 代码中给出了许多详细的中文注释，方便大家更好地理解代码(能力有限，可能有些代码或注释有误，望大家不吝赐教)

## 运行环境
python3.6、 transformers==2.1.1、pytorch==1.3.1

## 项目结构
- config:存放GPT2模型的参数的配置文件
- data
    - train.txt:默认的原始训练集文件，存放闲聊语料 
    - train_tokenized.txt:对原始训练语料进行顺序tokenize之后的文件，用于dialogue model的训练
    - train_mmi_tokenized.txt:对原始训练语料进行逆序tokenize之后的文件，用于mmi model的训练
- dialogue_model:存放对话生成的模型
- mmi_model:存放MMI模型(maximum mutual information scoring function)，用于预测P(Source|response)
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

## Dialogue Model
Dialogue Model是基于GPT2模型的生成模型，对每条训练数据进行"顺序"拼接，然后将其输入到网络中，进行训练(此处的"顺序"是相对于MMI Model的"逆序")

例如存在如下多轮闲聊训练数据,在训练Dialogue Model时，将上述训练数据进行如下拼接:**"[CLS]想看你的美照[SEP]亲我一口就给你看[SEP]我亲两口[SEP]讨厌人家拿小拳拳捶你胸口[SEP]"**。然后将上述拼接结果作为Dialogue Model的输入，对模型进行训练
```
想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口
```


## MMI Model
MMI Model的思想基于微软的论文[DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)

MMI Model也是一个基于GPT2的生成模型，将每条训练数据进行"逆序"拼接,然后输入到网络中。该模型主要用于计算Dialogue Model生成的所有候选response相对于dialogue history的loss。

训练时，将一条训练语料进行逆序拼接，如 **"[CLS]讨厌人家拿小拳拳捶你胸口[SEP]我亲两口[SEP]亲我一口就给你看[SEP]想看你的美照[SEP]"**，并作为MMI Model的输入进行训练
```
想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口
```

## response生成步骤
- 假设当前dialogue history=["你好","你好呀","你在干嘛呢"]
- 首先使用Dialogue Model根据dialogue history生成n个候选response:["在看电视","我在上课啊","人家在想你啊","我不知道"]
- 使用MMI Model将每个候选response分别与dialogue history进行逆序拼接，如 **"[CLS]在看电视[SEP]你在干嘛呢[SEP]你好呀[SEP]你好[SEP]"**
- 将上述拼接结果作为MMI Model的输入，计算每个response的loss
- 选择loss最小的response作为最终的结果进行回复


## 闲聊语料分享
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘【提取码:jk8d】](https://pan.baidu.com/s/1mkP59GyF9CZ8_r1F864GEQ) 或 [GoogleDrive](https://drive.google.com/file/d/1nEuew_KNpTMbyy7BO4c8bXMXN351RCPp/view?usp=sharing) |由作者[GaoQ1](https://github.com/GaoQ1)提供的比较高质量的闲聊数据集，整理出了50w个多轮对话的语料|

50w中文闲聊语料的内容样例如下:
```
谢谢你所做的一切
你开心就好
开心
嗯因为你的心里只有学习
某某某，还有你
这个某某某用的好

你们宿舍都是这么厉害的人吗
眼睛特别搞笑这土也不好捏但就是觉得挺可爱
特别可爱啊

今天好点了吗？
一天比一天严重
吃药不管用，去打一针。别拖着
```

## 模型分享
闲聊语料大小为67M，包含50w个多轮对话。使用该语料训练了两个模型dialogue_model与mmi_model

|模型 | 百度网盘 |GoogleDrive |模型描述|
|---------|--------|--------|--------|
|dialogue_model | [百度网盘【提取码:osi6】](https://pan.baidu.com/s/1qDZ24VKLBU9GKARX9Ev65g) | [GoogleDrive](https://drive.google.com/drive/folders/1Ogz3eapvtvdY4VUcY9AEwMbNRivLKhri?usp=sharing) |使用闲聊语料训练了40个epoch，最终loss在2.0左右，继续训练的话，loss应该还能继续下降。|
|mmi_model | [百度网盘【提取码:1j88】](https://pan.baidu.com/s/1ubXGuEvY8KmwEjIVTJVLww) | [GoogleDrive](https://drive.google.com/drive/folders/1oWgKXP6VG_sT_2VMrm0xL4uOqfYwzgUP?usp=sharing) |以dialogue_model作为预训练模型，使用上述闲聊语料，训练了40个epoch，最终loss在1.8-2.2之间，继续训练的话，loss也能继续下降。|

## 模型使用方法

把下载好的模型文件夹dialogue_model与mmi_model放在项目根目录下(否则需要通过--dialogue_model_path与--mmi_model_path参数指定对应模型的路径)，执行如下命令:
### 仅使用dialogue_model进行生成
``` bash
python interact.py --no_cuda(使用默认参数，不使用GPU。由于闲聊对话生成的内容长度不是很长，因此生成部分在CPU上跑速度也挺快的)
或
python interact.py --no_cuda --dialogue_model_path path_to_dialogue_model --max_history_len 5(自定义--max_history_len参数，即对话历史的长度)
或
python interact.py --no_cuda --dialogue_model_path path_to_dialogue_model --max_history_len 5 --topp 0.8 --topk 0(--topp为0到1之间的小数，用于调用Nucleus Sampling)
或
python interact.py --no_cuda --max_history_len 5 --topk 8(未指定--dialogue_model_path参数，默认为dialogue_model)
``` 
输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的sample.txt文件中

### 使用dialogue_model生成多个候选response，然后使用mmi_model选取互信息loss最小的response
interact_mmi.py的用法与interact.py类似
``` bash
python interact_mmi.py --no_cuda(使用默认的model路径)
或
python interact_mmi.py --no_cuda --batch_size 5(指定生成候选response的个数)
或
python interact_mmi.py --no_cuda --debug(debug模式，可以看到生成的所有候选response及其通过mmi_model的loss)
或
python interact_mmi.py --no_cuda --dialogue_model_path path_to_dialogue_model --mmi_model_path path_to_mmi_model(自定义模型路径)
``` 
输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的mmi_samples.txt文件中

更多的参数介绍，可直接看interact.py和interact_mmi.py中的setup_train_args()函数中的参数说明

## interact.py与interact_mmi.py的参数
执行interact.py时，可以尝试通过调整topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果。详细的参数描述可以查看interact.py的set_interact_args()函数

## 训练模型
在项目根目录下创建data文件夹，将原始训练语料命名为train.txt，存放在该目录下。train.txt的格式如下，每段闲聊之间间隔一行。

![avatar](image/chitchat_demo.png)

训练模型：
``` bash
# 若是训练mmi_model则需要指定--train_mmi参数；若是训练dialogue_model，则不需要指定--train_mmi参数

#训练dialogue_model
python train.py --epochs 30 --batch_size 8 --device 0,1 --raw(若要对原始训练语料进行tokenize，则要指定--raw参数。若要用GPU训练，则通过--device指定GPU)
或
python train.py --epochs 30 --batch_size 8 --no_cuda --raw(指定--no_cuda参数，则使用CPU训练，速度要慢得多)
或
python train.py --epochs 30 --batch_size 8 --no_cuda(若已经对原始语料进行tokenize，可以不用指定--raw，避免重复tokenize，节约时间)

#训练mmi_model,要指定--train_mmi参数
python train.py --epochs 30 --batch_size 8 --device 0,1 --raw --train_mmi(对原始训练语料进行逆序拼接，tokenize，并且训练mmi_model)
或
python train.py --epochs 30 --batch_size 8 --device 0,1 --train_mmi(若已经对原始训练语料tokenize，则直接训练mmi_model)
或
python train.py --epochs 30 --batch_size 8 --device 0,1 --train_mmi --pretrained_model path_to_pretrained_model(在与训练模型基础上继续训练)
```
更多的参数介绍，可直接看train.py中的setup_train_args()函数中的参数说明

## interact.py生成样例
#### Sample 1:
``` 
  user :你好啊
chatbot:你好，请问您是？
  user :我是你的主人啊 
chatbot:你是狗吗？
  user :你才是狗
chatbot:你不是狗我是你的小主人
  user :你这个大笨蛋
chatbot:你大笨蛋，你是想挨打了
  user :你全家都是大笨蛋
chatbot:我是你爸爸
  user :算了，不跟你吵了，你在干嘛呢
chatbot:我在上班啊，你还没下班呢啊你
  user :我周末要做作业啊
chatbot:不是才交了吗，早晚做完啊
  user :好多作业啊，还没做完呢
chatbot:我在抄，我妈让我周五交，还有事
  user :你就不怕被你妈妈骂吗
chatbot:不怕
  user :你周末有去哪里玩吗
chatbot:没时间啊，忙得很啊
  user :忙着干嘛呢
chatbot:不知道啊，我周末没有作业，我想睡觉，然后还得找人
``` 

#### Sample 2:
``` 
  user :你好啊
chatbot:你好我好大家好
  user :你在干嘛呢
chatbot:你不上课啊
  user :今天是周末啊，笨蛋
chatbot:我在学校呢
  user :我也是，周末还要做作业
chatbot:我还没写呢我们都还没写
  user :周末你有去哪里玩吗
chatbot:没有去南京
  user :南京有什么好玩的啊
chatbot:我们没去南京
  user :你刚刚又说你去了南京，你是不是耍我啊
chatbot:我刚刚才到南京
  user :你好笨啊
chatbot:我没有
  user :你是大笨蛋
chatbot:你才是大笨蛋
  user :你全家都是大笨蛋
chatbot:你个傻冒
  user :你个智障
chatbot:哈哈哈
``` 
#### Sample 3:
``` 
  user :在干嘛
chatbot:睡觉
  user :这么早就睡觉啦
chatbot:嗯啊你也早点睡啊
  user :我再玩一会儿手机
chatbot:早点休息晚安
  user :拜拜啦
chatbot:晚安晚安
  user :多喝热水
chatbot:哈哈哈
  user :多喝热水
chatbot:好的咯
``` 
#### Sample 4:
``` 
  user :在吗
chatbot:在
  user :我不开心
chatbot:怎么啦
  user :最近好累啊
chatbot:是挺累的
  user :作业好多啊
chatbot:还好还好
  user :我觉得我快要挂科了
chatbot:我也是啊，但是我觉得我快熬出头了
  user :不说了，我还有好多作业没写
chatbot:我也是啊好多啊
  user :那先这样，下次再聊，再见了
chatbot:好哒
``` 

#### Sample 5:
``` 
  user :你去过广州吗
chatbot:去过呀，但是没有找到工作呢
  user :那你去过北京吗
chatbot:没有啊
  user :北京有什么好玩的
chatbot:没有你想象的那么好
  user :是吗，我还没过北京
chatbot:那里我也不怎么熟，我去过北京
  user :北京天安门呢
chatbot:没去过，不过可以去玩玩
``` 

#### Sample 6:
``` 
  user :你好
chatbot:你好
  user :在干嘛呢
chatbot:在想你呢
  user :好害羞啊
chatbot:害羞什么啊
  user :你这样夸我，我觉得不好意思呀
chatbot:我夸你你还不好意思呢
  user :是啊，哈哈哈
chatbot:你是不是在考验我
  user :你怎么会这样想呢
chatbot:我说我在夸你，这么客气，
  user :你有睡午觉吗
chatbot:有啊
  user :我现在有点困呢
chatbot:我也是
  user :等下还要去上课，好烦啊
chatbot:我们不用上课的呀
``` 

## interact_mmi.py生成样例(debug模式下)
添加了MMI Model之后，效果确实提高了许多，可以看到，使用loss越小的response进行回复，对话越流畅

每个Sample为完整的一段对话，为了方便阅读，每轮对话之间加上了换行符。candidate response为dialogue model生成的候选response

需要指定--debug参数才会输出候选response及其loss
#### Sample 1:
```
user:你好
candidate response:
你好 loss:1.8774782419204712
你好！ loss:2.1092255115509033
你好 loss:1.8774782419204712
你好，朋友 loss:1.9210820198059082
你好！ loss:2.1092255115509033
chatbot:你好

user:在干嘛呢
candidate response:
上课玩手机 loss:1.8411542177200317
上班呢 loss:1.8623629808425903
上班啊 loss:1.8383764028549194
刚吃完饭，你呢，我今天吃饱了没事，后天出去吃宵夜， loss:2.471815586090088
刚刚搬家 loss:2.341609239578247
chatbot:上班啊

user:周末要一起出去玩吗
candidate response:
可以呀 loss:2.0150623321533203
可以呀 loss:2.0150623321533203
可以，你还是做老师吗 loss:2.2639517784118652
可以啊 loss:1.9695569276809692
周末要 loss:2.1701855659484863
chatbot:可以啊

user:我好想你啊
candidate response:
我也好想你啊 loss:1.8660272359848022
那我们就可以一起出来玩呀 loss:2.046417236328125
我也好想你啊 loss:1.8660272359848022
我也很久没出去玩啦 loss:2.17940354347229
来找我吧 loss:2.14339542388916
chatbot:我也好想你啊

user:爱你么么哒
candidate response:
么么哒 loss:1.6875463724136353
好想你可是也不着急了 loss:2.2054319381713867
么么哒 loss:1.6875463724136353
你会爱上新花园的人吗 loss:2.521526575088501
我也爱你 loss:1.6839145421981812
chatbot:我也爱你

user:你喜欢我吗
candidate response:
那必须是啊 loss:1.545378565788269
喜欢你啊 loss:1.4527238607406616
喜欢喜欢 loss:1.4971727132797241
还行吧 loss:1.5946317911148071
你是我的小公举么么哒你还不睡觉啊 loss:1.6506415605545044
chatbot:喜欢你啊
```

#### Sample 2
```
user:亲爱的在干嘛呢
candidate response:
刚看了一个小片段子，好难过 loss:2.541116714477539
你猜呀 loss:2.1126084327697754
亲爱的休息啊 loss:2.073425054550171
在家里玩儿 loss:1.9639413356781006
我在等你啊 loss:1.9869526624679565
chatbot:在家里玩儿

user:作业写完了吗
candidate response:
没啊，你呢 loss:2.01458477973938
没有今天刚写完 loss:2.0036869049072266
没，今天刚写完 loss:2.0473012924194336
没有 loss:1.9207587242126465
没呢，明天上午没课 loss:1.943585753440857
chatbot:没有

user:还不快滚去写作业
candidate response:
我们还没写呢 loss:2.271885871887207
我还没写呢 loss:2.1460421085357666
哈哈哈我还没 loss:2.2205970287323
去死 loss:2.469320058822632
我们还要写作业 loss:2.247976779937744
chatbot:我还没写呢

user:你个大笨蛋
candidate response:
我就说 loss:2.052217483520508
你个大傻纸 loss:1.9332027435302734
可以 loss:2.0493569374084473
你才是 loss:1.8672529458999634
我们五月五号才放寒假 loss:2.1910266876220703
chatbot:你才是

user:你是傻子吧
candidate response:
是你是你 loss:2.3064982891082764
你是大傻 loss:2.4247941970825195
好好上课 loss:2.4157748222351074
你是傻子你说谁是傻子 loss:2.1408958435058594
是你才是 loss:2.4862639904022217
chatbot:你是傻子你说谁是傻子

user:说的就是你，怎样
candidate response:
傻子，你才傻 loss:2.2158799171447754
我不傻 loss:2.188838005065918
傻的跟你一样 loss:2.3169684410095215
那你是聪明的小孩，你想怎样？ loss:2.2750229835510254
你才傻 loss:2.262368679046631
chatbot:我不傻

user:不和你玩了
candidate response:
你怎么那么傻 loss:2.126516819000244
我是傻子 loss:2.044743061065674
我不和你说话了 loss:2.107841730117798
不要 loss:2.124176502227783
我不和你玩你说我傻 loss:2.1581454277038574
chatbot:我是傻子
```


## 不足之处
没有对chatbot生成的response的内容进行检测，有时会生成一些敏感、略带玩笑性质的辱骂内容

## Future Work
对生成内容进行控制

## Reference
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)




