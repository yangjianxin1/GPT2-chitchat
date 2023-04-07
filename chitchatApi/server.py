import flask
from interact import *
from flask import Flask, session
from flask_cors import cross_origin #需单独安装

args = set_args()
app = flask.Flask(__name__)
app.secret_key = '123456'
model = None
model2=None
tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
# history=[]
model_path1="./model1"
model_path2="./model2"
# 加载模型
def load_model():
    global model
    global model2
    # 模型路径
    model = GPT2LMHeadModel.from_pretrained(model_path1)
    model.eval()
    # 同模型路径
    model2 = GPT2LMHeadModel.from_pretrained(model_path2)
    model2.eval()


@app.route("/predict", methods=["post"])
@cross_origin()  #设置跨域
def predict():
    if flask.request.method=="POST":
        msg = flask.request.form
        message = msg['message']
        data={"errno": 0, "msg": message}
        return flask.jsonify(data)


@app.route("/model1Reply", methods=["post"])
@cross_origin()
def model_reply():
    global model
    # global history
    global tokenizer
    if flask.request.method == "POST":
        msg = flask.request.form
        print("msg:", msg)
        text = msg['message']
        if len(text)>25:
            data = {"errno": 201, "msg": "输入字符过长(长度限制25）"}
            return flask.jsonify(data)
       # 可以通过设置action，改变对模型的使用方法
        action = msg['action']
        if action=='1':
            print("模式1")
            history = session.get('history1')
            if history == None:
                history = []
        else:
            print("模式2")
            history = session.get('history2')
            if history == None:
                history = []
        print(text)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        history.append(text_ids)
        input_ids = [tokenizer.cls_token_id]
        for history_id, history_utr in enumerate(history[-args.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0)
        response = []  # 根据context，生成的response
        # 最多生成max_len个token
        for _ in range(args.max_len):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(response):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        history.append(response)
        if action=='1':
            session['history1'] = history
        else:
            session['history2'] = history
        text = tokenizer.convert_ids_to_tokens(response)
        text = "".join(text)
        data = {"errno":0, "msg": text}
        print(data)
        return flask.jsonify(data)


@app.route("/model2Reply", methods=["post"])
@cross_origin()
def model2_reply():
    global model2
    # global history
    global tokenizer
    if flask.request.method == "POST":
        msg = flask.request.form
        text = msg['message']
        if len(text)>25:
            data = {"errno": 201, "msg": "输入字符过长(长度限制25）"}
            return flask.jsonify(data)
        action = msg['action']
        if action == '1':
            print("模式1")
            history = session.get('history1')
            if history == None:
                history = []
        else:
            print("模式2")
            history = session.get('history2')
            if history == None:
                history = []
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        history.append(text_ids)
        input_ids = [tokenizer.cls_token_id]
        for history_id, history_utr in enumerate(history[-args.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0)
        response = []  # 根据context，生成的response
        # 最多生成max_len个token
        for _ in range(args.max_len):
            outputs = model2(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(response):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        history.append(response)
        if action=='1':
            session['history1'] = history
        else:
            session['history2'] = history
        text = tokenizer.convert_ids_to_tokens(response)
        text = "".join(text)
        data = {"errno": 0, "msg": text}
        return flask.jsonify(data)


@app.route("/clear_history", methods=["post"])#退出时清除所有session
@cross_origin()
def logout():
    session.clear()
    data = {"errno": 0, "msg": "清除成功"}
    return flask.jsonify(data)


@app.route("/clear_history1", methods=["post"])#退出时清除所有session
@cross_origin()
def clear_history1():
    history = session.get('history1')
    print("开始清除history1", history)
    if history != None:
        print('debug：清除history1')
        session.pop('history1')
    data = {"errno": 0, "msg": '清除history1成功'}
    print(session.get('history1'))
    return flask.jsonify(data)


@app.route("/clear_history2", methods=["post"])#退出时清除所有session
@cross_origin()
def clear_history2():
    print(session.get('history2'))
    if session.get('history2')!=None:
        session.pop('history2')
    data = {"errno": 0, "msg": '清除history2成功'}
    print(session.get('history2'))
    return flask.jsonify(data)


if __name__ == '__main__':
    from wsgiref.simple_server import make_server

    KMP_ABORT_IF_NO_IRML = False
    load_model()
    # server = make_server('127.0.0.1', 5000, app)
    # server.serve_forever()
    app.run()