#### API简介：

./chitchatApi/serve.py会同时加载两个模型，实现两个模型和人的共同对话

#### 使用准备

下载模型放在路径 ./chitchatApi下

> 参数model_path1和model_path2对应着模型路径，如果只有一个模型可以设置为相同路径



#### Api接口解释：

##### 1.对话

```python
@app.route("/model1Reply", methods=["post"])#模型一接口
@app.route("/model2Reply", methods=["post"])#模型二接口
```

> 需携带两个参数：
>
> {message:"",action:"" }
>
> message为需要被回应的对话， action置1即可

response：

```
data = {"errno":0, "msg": text}
```

##### 2.删除历史记录：

>```
>@app.route("/clear_history", methods=["post"])#退出时清除所有session
>```