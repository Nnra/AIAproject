# 目錄架構和檔案說明 
Last Update: 2020-04-19
```
final
├── Ravdess_model
    ├── Emotion_Voice_Detection_Model.h5 音訊情緒辨識 Model
├── face 存放臉部辨識模型的資料和程式
    ├── models
        ├── base_emotion_classification_model.hdf5 臉部情緒分類model baseline版本0
        ├── emotion_classificator_0.hdf5 臉部情緒分類model ver.0
        ├── emotion_classificator_base_1.hdf5 臉部情緒分類model baseline 版本1
        ├── emotion_classificator_base_2.hdf5 臉部情緒分類model baseline 版本2
        ├── emotion_classificator_base_3.hdf5 臉部情緒分類model baseline 版本3
        ├── opencv_face_detector_unit8.pb open 定義節點網路架構，用來封存權重值的pb檔
        ├── opencv_face_detector.pbtxt 定義節點網路架構，用來封存權重值的pbtxt檔
        ├── shape_predictor_68_face_landmarks.dat 將影片壓縮處理的dat檔
    ├── preprocessing 放入model之前的處理，將mp4轉為frames再轉為pickle
        ├── Frames2Pickle.ipynb 將frames轉換成pickle的ipynb檔
        ├── frames2pickle.py 將frames轉換成pickle的py檔
        ├── video2frames.py 將video轉換成frames的py檔
    ├── .nfs00000000004c00180000000e 檔案修改自動生成檔
    ├── emotion_detector.py 使用model預測臉部情緒
    ├── face_emotion.py 使用model預測臉部情緒，搭配counter得到所有情緒總和最大值
    ├── function.py 用來得到預測的臉部情緒種類和情緒佔比
    ├── predict-Copy1.ipynb 對單一影片進行臉部情緒辨識用程式
    ├── predict.ipynb 對影片進行臉部情緒辨識主程式
    ├── requirements.txt 說明每個工具採用版本
    ├── train.ipynb 臉部情緒辨識模型訓練主程式
├── joblib 用來存放訓練資料wav檔經過前處理後的檔案
    ├── X.joblib 訓練資料經前處理的array
    ├── y.joblib 訓練資料經前處理的emotion
├── 20200327_Lim_enhance_trim_epooch_200.ipynb 音訊情緒辨識模型調參程式
├── README.md 介紹目錄架構和檔案說明
├── dataaugmentation.py 資料前處理
├── get_content_from_youtube.ipynb 擷取youtube影片程式
├── model_1.ipynb 音訊情緒辨識模型主程式
├── youtube_to_wav.ipynb 將youtube影片轉為wav檔程式
```

# 專題處理方法說明
## 問題描述與專題目的

#### 專題目的：

辨識音源或影片中講者聲音和表情所傳達的情緒， 依據此情緒提供對應的曲風類型或音效，可以做為影片後製的配樂或音效。透過我們的系統可以幫助使用者在影片後製階段，辨識講者語音、表情情緒以及減少尋找合適配樂的時間。

參考資料
--- 
#### 相關文獻
- Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.
- Machine Recognition of Music Emotion: A Review http://mac.citi.sinica.edu.tw/~yang/pub/yang12tist.pdf
- SoundNet: Learning Sound Representations from Unlabeled Video https://arxiv.org/pdf/1610.09001.pdf
- MULTI-LABEL CLASSIFICATION OF MUSIC INTO EMOTIONS http://lpis.csd.auth.gr/publications/tsoumakas-ismir08.pdf

---
#### GitHub
- Emotion-Classification-Ravdess : 
  https://github.com/marcogdepinto/Emotion-Classification-Ravdess
- Emotion_classification_Ravdess :   https://github.com/sunniee/Emotion_classification_Ravdess

--- 
## 資料集說明
### 1. Dataset Used for AI training: RAVDESS
- The database is gender balanced consisting of 24 professional actors, vocalizing lexically-matched statements in a neutral North American accent. 
- Speech includes neutral, calm, happy, sad, angry, fearful, surprise, and disgust emotions
- Song contains neutral, calm, happy, sad, angry, and fearful emotions
- The dataset can be divided into two types: sound-only (.wav format) and video clips (.mp4 format)
- There are total of 4948 samples of voice and song clips (including videos)from the RAVDESS database
- Each expression is produced at two levels of emotional intensity
- The recording method and expression is shown below:
![](https://i.imgur.com/qRdrMIJ.png)


> Reference: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. 




Example of RAVDESS Speech and Songs
| Speeches | Songs | 
| -------- | -------- | 
| <a href="http://www.youtube.com/watch?feature=player_embedded&v=Y7OQoNEu3dY" target="_blank"><img src="http://img.youtube.com/vi/Y7OQoNEu3dY/0.jpg" alt="Speeches" width="240" height="180" border="20" /></a>     | <a href="http://www.youtube.com/watch?feature=player_embedded&v=XQkmH4oYZkg" target="_blank"><img src="http://img.youtube.com/vi/XQkmH4oYZkg/0.jpg" alt="Songs" width="240" height="180" border="20" /></a>     | 


How to handle RAVDESS dataset for AI Training (Voice)

![](https://i.imgur.com/sWyGGlK.png)

How to handle RAVDESS dataset for AI Training of face
![](https://i.imgur.com/p8YGjil.png)

### 2. Dataset used for Music Recommendation: Lyrics Mood Classification (LMC)

- The database is handled by UC Berkeley Masters of Information & Data Science
- The project plans to compare the accuracy of deep learning vs traditional machine learning approaches for classifying the mood of songs, using lyrics (歌詞)  as features.
- Made use of the Million Song Dataset (MSD) and its companion datasets from Last.fm and MusixMatch.
- The project scrap, index, and labeling the music lyrics before applying to their CNN deep learning project.
- Mood/Emotion is divided into 18 categories: aggression, anger, brooding, calm, cheerful, confident, depressed, desire, earnest, excitement, grief, happy, pessimism, romantic, sad, and unknown.
- They make the labeled music song database available for free download 

> Project Link: https://github.com/workmanjack/lyric-mood-classification
Reference: https://github.com/workmanjack/lyric-mood-classification/blob/master/report/lyric-mood-classification-with-deep-learning.pdf

How to handle LMC dataset 

<img src="https://i.imgur.com/0k40NCa.png" width=100%>


Emotion Differences between RAVDESS and LMC
- The emotion classification of RAVDESS and LMC is significant different as shown below.
- LMC has more complex emotions than the RAVDESS baed on Cambridge dictionary definition.
- LMC emotion and songs needed to be modified into simpler emotion, and fit RAVDESS dataset.

| RAVDESS Emotion | 英中翻譯 (劍橋字典) |  
| ------------  | ---------------------- |  
| Angry         | 發怒的，憤怒的，生氣的    |  
| Fearful       | 恐懼的；擔心的；憂慮的     |  
| Disgusted     | 反感的，厭惡的，憎惡的     |  
| Neutral       | 中立的，不偏不倚的     |  
| Calm          | 冷靜的，鎮靜的     |  
| Happy         | 幸福的，滿意的，快樂的     |  
| Surprised     | 意外的，驚訝的，詫異的     |  
| Sad           | 傷心的，悲哀的；令人難過的，令人遺憾的 |  

| LMC Emotion (Total Songs) | 英中翻譯 (劍橋字典)|  
| -------- | -------- |  
| Aggression (499)     | 侵略；侵犯；攻擊；挑釁     |  
| Anger (1046)         | 怒，憤怒；怒火            |  
| Angst (251)          | 焦慮，煩憂               |  
| Brooding (169)       | 令人憂心忡忡的            |  
| Calm (2830)          | 冷靜的，鎮靜的            |  
| Cheerful (218)       | 高興的，快樂的；興高采烈的  |  
| Confident (59)       | 自信的；有信心的；確信的；有把握的；信任的    |  
| Depressed (2736)     |憂鬱的，消沉的，沮喪的 |
|Desire (130)|渴望，希望，想要|


| LMC Emotion (Total songs) | 英中翻譯 (劍橋字典)|
|--|--|
|Earnest (119)|認真的；有決心的；鄭重其事的；誠摯的|
|Excitement (169)|激動，興奮；令人興奮的事情|
|Grief (390)|悲痛，悲傷，悲哀|
|Happy (3242)|幸福的，滿意的，快樂的|
|Pessimism (6)|悲觀情緒；悲觀主義|
|Romantic (2450)|愛情的，情愛的|
|Sad (7217)|傷心的，悲哀的；令人難過的，令人遺憾的|
|Upbeat (4235)|樂觀的；快樂的；積極向上的|


Convert LMC dataset into RAVDESS format
- We need to convert the LMC emotion label into RAVDESS emotion label to create music recommendation system
<img src="https://i.imgur.com/06h8ns6.png" width=50%>

:::info
Note
- If LMC songs are allocated into 2 different emotions, the odd number of music from the list will be allocated to 1 emotion, while the even number of music from the list will be allocated to another emotion

- If LMC songs have more music than the RAVDESS emotions, the even number of the music from the list will be picked for the emotion

:::

Music Recommendation Table For Applications
| Emotion Label|Total Music/Youtube videos|
|---|---|
|Angry|772|
|Happy|1000|
|Sad|1000|
|Neutral|1000|
|Calm|1000|
|Surprised|169|
|Disgusted|773|
|Fearful|1000|
|Total music|6714|

## Model
### 音頻擷取

- Short-time fourier transform​
```=python
X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
```
<img src="https://i.imgur.com/iXcz3mR.png" width=50%>

- Hamming windows​
```=python
data, index = librosa.effects.trim(data, top_db=30, frame_length=1024, hop_length=512)
```

<img src="https://i.imgur.com/B8o9ytH.png" width=50%>


- Mel-scale Filter Bank​
<img src="https://i.imgur.com/AX0tv4J.png" width=50%>

-  Logarithmic Operation and IDFT​
<img src="https://i.imgur.com/jWrQ68l.png" width=50%>

-  MFCC​
```=python
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T,axis=0)
```

<img src="https://i.imgur.com/PmOG6fn.png" width=50%>

- librosa.load

```=python
librosa.core.load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_fast')
```

- librosa.effects.trim

```=python
librosa.effects.trim(y, top_db=30, ref=<function amax at 0x7fa274a61d90>, frame_length=1024, hop_length=512)
```

- librosa.feature.mfcc

```=python
librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=100, dct_type=2, norm='ortho', lifter=0, **kwargs)
```

### 模型 - 模型說明-model 1 結構與方法(改良後)


<img src="https://i.imgur.com/f2L7aVu.png" width=80&></img>

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 100, 128)          768       
_________________________________________________________________
batch_normalization_1 (Batch (None, 100, 128)          512       
_________________________________________________________________
activation_1 (Activation)    (None, 100, 128)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 100, 128)          82048     
_________________________________________________________________
batch_normalization_2 (Batch (None, 100, 128)          512       
_________________________________________________________________
activation_2 (Activation)    (None, 100, 128)          0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 12, 128)           0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 12, 128)           82048     
_________________________________________________________________
batch_normalization_3 (Batch (None, 12, 128)           512       
_________________________________________________________________
activation_3 (Activation)    (None, 12, 128)           0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 12, 128)           82048     
_________________________________________________________________
batch_normalization_4 (Batch (None, 12, 128)           512       
_________________________________________________________________
activation_4 (Activation)    (None, 12, 128)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1536)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 12296     
_________________________________________________________________
activation_5 (Activation)    (None, 8)                 0         
=================================================================
Total params: 261,256
Trainable params: 260,232
Non-trainable params: 1,024
_________________________________________________________________

```

### Siamens Network Description

<img src="https://i.imgur.com/oX3rRGb.png" width=60%></img>


### model 1 ：Voice Emotion Predict Architecture

<img src="https://i.imgur.com/NvzCFEi.png" width=80%></img>


### model 2 ：Facial Emotion Predict Architecture

<img src="https://i.imgur.com/YcQQhaQ.png" width=80%></img>

> Reference:
https://github.com/sunniee/Emotion_classification_Ravdess

### model 2 ：Facial Emotion Predict Description

<img src ="https://i.imgur.com/3DqfZNV.png" width=50%></img>

### model 2 ：Facial Emotion Predict Process

<img src="https://i.imgur.com/10VdSbU.png" width=80%></img>

## 系統
### System Architecture

<img src="https://i.imgur.com/tSS6ywo.png" width=80%></img>


### Architecture - Web Data Flow

<img src="https://i.imgur.com/8fBwFpX.png" width=80%></img>

### How to pick music list and record behavior ?

```
After prediction, get emption from model 1 and model 2.
Use predicted emption of model 1 to choose 5 records with the same labeled music emotion in database, and do the same thing in model 2.
Show 10 music records in web page, users can review the music, and when the users download the music, the behavior will be saved included music youtube id, current users’ emotion and download time. 
```

### Architecture - Web Stack

<img src="https://i.imgur.com/GeowksP.png" width=80%></img>


## DEMO

![Link to DEMO](https://obscure-ravine-38834.herokuapp.com) 





# 專題程式架構說明

# The project

The scope of this project is to create a classifier to predict the emotions of the speaker starting from an audio file.
And randomly pick five music with the same emotion as the user.

**Dataset**

For this task, I have used 4948 samples from the RAVDESS dataset (see below to know more about the data).

The samples comes from:

- Audio-only files;

- Video + audio files: I have extracted the audio from each file using the script **Mp4ToWav.py** that you can find in the main directory of the project.

The classes we are trying to predict are the following: (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised)

# Actual metrics after the application of a Neural Network to this dataset

**Model summary**

![Link to loss](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/modelSummary.png) 

**Loss and accuracy plots**

![Link to loss](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/loss.png) 

![Link to accuracy](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/accuracy.png)

**Classification report**

![Link do classification report](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/classificationReportUpdated.png)

**Confusion matrix**

![Link do classification report](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/confusionMatrix.png)

**結果 - 結論**

我們專題研究的成果顯示語音情緒辨識資料集和音樂情緒辨識資料集，可以應用於由情感查詢音樂的音樂資訊檢索系統中，並且搭配臉部情緒辨識資料集，可以在情感辨識的準確度上有顯著提升，因此基於影片語音和人物臉部情緒作為音樂推薦的系統，在影片後製挑選配樂階段是有幫助的。

**Future Work**

系統已紀錄音樂被使用者下載的次數到資料庫中，同時會紀錄使用者的被預測出的情緒、時間、下載的音樂名稱及 youtube id。有了這些資訊可以訓練第3個模型，第3個模型將用來更準確的做出音樂清單的推薦。




# Tools and languages used

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [Google Colab](https://colab.research.google.com/)
- [Google Drive](https://drive.google.com)
- [Jupyter Notebook](http://jupyter.org/)

# Try it!

- Install Tensorflow, Librosa, Keras, Numpy.

- `git clone https://github.com/marcogdepinto/Emotion-Classification-Ravdess.git`

- Run the file `LivePredictions.py` changing the `PATH` and `FILE` to the local path in which you have downloaded the example file `01-01-01-01-01-01-01.wav` (or any other file of the RAVDESS dataset from their website) and the model `Emotion_Voice_Detection_Model.h5`

# Include this work in a web API

I am actually working on a new project using Django to serve this model in a web application: more info at https://github.com/marcogdepinto/Django-Emotion-Classification-Ravdess-API

# About the RAVDESS dataset

**Download**

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) can be downloaded free of charge at https://zenodo.org/record/1188976. 

**Construction and Validation**

Construction and validation of the RAVDESS is described in our paper: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

The RAVDESS contains 7356 files. Each file was rated 10 times on emotional validity, intensity, and genuineness. Ratings were provided by 247 individuals who were characteristic of untrained adult research participants from North America. A further set of 72 participants provided test-retest data. High levels of emotional validity, interrater reliability, and test-retest intrarater reliability were reported. Validation data is open-access, and can be downloaded along with our paper from PLOS ONE.

**Description**

The dataset contains the complete set of 7356 RAVDESS files (total size: 24.8 GB). Each of the 24 actors consists of three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  Note, there are no song files for Actor_18.

**License information**

“The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)” by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

**File naming convention**

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers 

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
- Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

**Filename example: 02-01-06-01-02-01-12.mp4**

- Video-only (02)
- Speech (01)
- Fearful (06)
- Normal intensity (01)
- Statement “dogs” (02)
- 1st Repetition (01)
- 12th Actor (12)
- Female, as the actor ID number is even.
