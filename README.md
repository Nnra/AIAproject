## 問題描述與專題目的

#### 專題目的：

辨識音源或影片中講者聲音和表情所傳達的情緒， 依據此情緒提供對應的曲風類型或音效，可以做為影片後製的配樂或音效。透過我們的系統可以幫助使用者在影片後製階段，辨識講者語音、表情情緒以及減少尋找合適配樂的時間。

#### 問題描述：
在近幾年語音情緒辨識和音樂情緒辨識領域增加許多已標註的資料集，然而因兩者資料集所採用的情緒特徵不一致，導致難以直接為影片片段推薦相同情緒音樂，讓影片後製費時費力，我們欲建立基於語音和臉部情緒辨識的音樂推薦系統解決此問題。

#### 問題價值：
- 提高語音情緒辨識準確度
- 未來搭配 chat bot 可用於音樂搜尋
- 未來搭配歌詞情緒分析，可判斷音樂所適合的影片場景

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

