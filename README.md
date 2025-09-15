# index-tts-lora
本项目基于 Bilibili 的 [index-tts](https://github.com/index-tts/index-tts) ，提供 **LoRA 单说话人 / 多说话人** 的微调方案，用于提升精品说话人合成音频的 **韵律和自然度**。

### 训练与推理

#### 1. 音频 token 与 speaker condition 提取

```shell
# 提取 token 和 speaker condition
python tools/extract_codec.py --audio_list ${audio_list} --extract_condition

# audio_list 格式：音频路径 + 文本，以 \t 分隔
/path/to/audio.wav 小朋友们，大家好，我是凯叔，今天我们讲一个龟兔赛跑的故事。
```

提取完成后，会在 `finetune_data/processed_data/` 目录下生成对应文件夹及 `speaker_info.json` 文件。例如：

```shell
[
    {
        "speaker": "kaishu_30min",
        "avg_duration": 6.6729,
        "sample_num": 270,
        "total_duration_in_seconds": 1801.696,
        "total_duration_in_minutes": 30.028,
        "total_duration_in_hours": 0.500,
        "train_jsonl": "/path/to/kaishu_30min/metadata_train.jsonl",
        "valid_jsonl": "/path/to/kaishu_30min/metadata_valid.jsonl",
        "medoid_condition": "/path/to/kaishu_30min/medoid_condition.npy"
    }
]
```

#### 2. 训练

```shell
python train.py
```

#### 3. 推理

```
python indextts/infer.py
```
### 微调结果

本次实验数据来自 **凯叔讲故事** 的纯中文音频，总时长约 **30 分钟**，共 **270 条音频**。数据划分为 **训练集 244 条**、**验证集 26 条**。需要注意的是，文本是通过 ASR 和标点模型自动生成的，未经过人工校对，因此存在一定错误率。

训练样音如下，`他上了马车，来到了皇宫之中。`：[kaishu_train_01.wav](https://github.com/user-attachments/files/22336605/kaishu_train_01.wav)

#### 1. 音频合成效果

|                        合成文本                              | 合成音频                                           |
| ------------------------------------------------------------ | -------------------------------------------------- |
| 老宅的钟表停在午夜三点，灰尘中浮现一串陌生脚印。侦探蹲下身，发现地板缝隙里藏着一枚带血的戒指。 | [kaishu_cn_1.wav](https://github.com/user-attachments/files/22336613/kaishu_cn_1.wav)|
| 月光下，南瓜突然长出笑脸，藤蔓扭动着推开花园栅栏。小女孩踮起脚，听见蘑菇在哼唱古老的摇篮曲。 |     [kaishu_cn_2.wav](https://github.com/user-attachments/files/22336616/kaishu_cn_2.wav)|
| 那么Java里面中级还要学，M以及到外部前端的应用系统开发，要学到Java Script的数据库，要学做动态的网站。 |     [kaishu_cn_en_mix_1.wav](https://github.com/user-attachments/files/22336625/kaishu_cn_en_mix_1.wav) |
| 这份 financial report 详细分析了公司在过去一个季度的 revenue performance 和 expenditure trends。 |  [kaishu_cn_en_mix_2.wav](https://github.com/user-attachments/files/22336633/kaishu_cn_en_mix_2.wav) |
| 上山下山上一山，下一山，跑了三里三米三，登了一座大高山，山高海拔三百三。上了山，大声喊：我比山高三尺三。 |   [kaishu_raokouling.wav](https://github.com/user-attachments/files/22336634/kaishu_raokouling.wav)  |
| A thin man lies against the side of the street with his shirt and a shoe off and bags nearby. |  [kaishu_en_1.wav](https://github.com/user-attachments/files/22336636/kaishu_en_1.wav)|
| As research continued, the protective effect of fluoride against dental decay was demonstrated. |     [kaishu_en_2.wav](https://github.com/user-attachments/files/22336638/kaishu_en_2.wav)|

#### 2. 模型精度测试

<img width="1182" height="261" alt="image" src="https://github.com/user-attachments/assets/fb86938d-95d9-4b10-9588-2de1e43b51d1" />

### 感谢

[index-tts](https://github.com/index-tts/index-tts)

[finetune-index-tts](https://github.com/yrom/finetune-index-tts)
