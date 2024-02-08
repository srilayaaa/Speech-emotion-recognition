#importing
import pandas as pad
import numpy as nup
import glob
import soundfile
import os
import sys
import librosa
import seaborn as sbn
import matplotlib.pyplot as mplt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from IPython.display import Audio
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call dr
RavdessData = "/content/drive/MyDrive/Dataset1/"
ravdessDirectoryList = os.listdir(RavdessData)
fileEmotion = []
filePath = []
for dir in ravdessDirectoryList:
    actor = os.listdir(RavdessData+dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        fileEmotion.append(int(part[2]))
        filePath.append(RavdessData + dir+'/'+file)
        emotion_df = pad.DataFrame(fileEmotion,columns=['Emotions'])
        path_df = pad.DataFrame(filePath, columns=['Path'])
Ravdess_df = pad.concat([emotion_df,path_df],axis=1)
Ravdess df.Emotions.replace({1:'neutral',2:'calm',3:'happy',4:'sad',5:'ang'})
Ravdess_df.head()
dataPath = pad.concat([Ravdess_df],axis=0)
dataPath.to_csv("data_path.csv",index=False)
dataPath.head()

mplt.title('Count of Emotions', size=16)
sbn.countplot(dataPath.Emotions)
mplt.ylabel('Count',size=12)
sbn.despine(top=True,right=True,left=False,bottom=False)
mplt.show()
import librosa.display
def createWaveplot(data,sr,e):
    mplt.figure(figsize=(10,3))
    mplt.title("Waveplot for audio with {} emotion".format(e),size=15)
    librosa.display.waveplot(data,sr=sr)
    mplt.show()
def createSpectrogram(data,sr,e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    mplt.figure(figsize=(12,3))
    mplt.title('Spectrogram for audio with {} emotion'.format(e),size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time',y_axis='hz')
    mplt.colorbar()
# waveplot for emotion fear
emotion = 'fear'
path = nup.array(dataPath.Path[dataPath.Emotions==emotion])[2]
data,samplingRate = librosa.load(path)
createWaveplot(data, samplingRate, emotion)
createSpectrogram(data, samplingRate, emotion)
Audio(path)
# waveplot for emotion angry
emotion = 'angry'
path = nup.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data,samplingRate = librosa.load(path)
createWaveplot(data, samplingRate, emotion)
createSpectrogram(data, samplingRate, emotion)
Audio(path)
# waveplot for emotion happy:
emotion = 'happy'
path = nup.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data,samplingRate = librosa.load(path)
createWaveplot(data, samplingRate, emotion)
createSpectrogram(data, samplingRate, emotion)
Audio(path)
def extractFeature(fileName, mfcc, chroma, mel):
    with soundfile.SoundFile(fileName) as soundFile:
    audio = soundFile.read(dtype="float32")
    sample_rate = soundFile.samplerate
    if chroma:
        stft=nup.abs(librosa.stft(audio))
        result=nup.array([])
    if mfcc:
        mfccs=nup.mean(librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40).T, axis=0
        result=nup.hstack((result, mfccs))
    if chroma:
        chroma=nup.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=nup.hstack((result, chroma))
    if mel:
        mel=nup.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
        result=nup.hstack((result, mel))
    return result
#defining dictionaries to hold no of emotions
emotions={
'01':'neutral',
'02':'calm',
'03':'happy',
'04':'sad',
'05':'angry',
'06':'fearful',
'07':'disgust',
'08':'surprised' }
                       
observedEmotions=['angry', 'calm', 'happy', 'fearful', 'disgust']
def loadData(test_size=0.2):
    x,y = [],[]
    for file in glob.glob("/content/drive/MyDrive/Dataset//Actor_*//*.wav"):
    fileName=os.path.basename(file)
    emotion1 = emotions[fileName.split("-")[2]]
    if emotion1 not in observedEmotions:
        continue
    feature = extractFeature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    y.append(emotion1)
    final_dataset = train_test_split(nup.array(x), y, test_size=test_size, random_state
    return final_dataset
#splitting dataset
xTrain,xTest,yTrain,yTest=loadData(test_size=0.23)
#Getting the shape of the training and testing datasets
print(xTrain.shape[0],xTest.shape[0])

#Getting the number of features extracted
print(f'Features extracted: {xTrain.shape[1]}')
#Initialize the MultiLayer-Perception Classifier
model = MLPClassifier(early_stopping=True,alpha=0.01,batch_size=256,epsilon=1e-08,hid
#Training the model
model.fit(xTrain,yTrain)
MLPClassifier(alpha=0.01, batch_size=256, early_stopping=True,hidden_layer_sizes=(300,), learning_rate='adaptive',max_iter=400)
#Predicting for the test set
expected_Of_y = yTest
y_pred =model.predict(xTest)
print(metrics.confusion_matrix(expected_Of_y, y_pred))

print(classification_report(yTest, y_pred))
 
test_accuracy=accuracy_score(y_true=yTest,y_pred=y_pred)
train_pred=model.predict(xTrain)
# printing training accuracy
train_accuracy=accuracy_score(y_true=yTrain,y_pred=train_pred)
print("accuracy: {:.2f}%".format(train_accuracy*136))

#Calculating the accuracy of our model
accuracy =accuracy_score(y_true=yTest,y_pred=y_pred)
#printing the accuracy
print("accuracy: {:.1f}%".format(accuracy*149))

mplt.plot(model.validation_scores_, color='green', alpha=0.8)
mplt.title("Accuracy over iterations", fontsize=14)
mplt.xlabel('Iterations')
mplt.show()