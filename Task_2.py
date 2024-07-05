import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import LearningRateScheduler
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

class ocr_rec:
    def __init__(self,csv_path,image_path):
        self.csv_path=csv_path
        self.image_path=image_path

        self.lab=LabelEncoder()
        self.imageData=None
        self.labels=None
        self.model=None

        datacsv=pd.read_csv(csv_path)
        self.dataArray=datacsv.to_numpy()
        imglabels=self.dataArray[1:,1]
        self.labels=self.lab.fit_transform(imglabels)
        self.labels=to_categorical(self.labels)

    def save_data(self):
        imglabels=self.dataArray[1:,1]
        rows,cols=self.dataArray.shape
        imageData=[]
        for i in self.dataArray[1:,0]:
            path=os.path.join(self.image_path,i)
            image=load_img(path,target_size=(28,28),color_mode='grayscale')
            image=img_to_array(image)
            image=image/255.0
            imageData.append(image)
        self.imageData=np.array(imageData)
        np.save("imgpixels.npy", self.imageData)


    def load_data(self,npy_name):
        self.imageData=np.load(f'{npy_name}')

    def train_model(self,epochs=10,lr=0.001,decay=0.5):
        def lr_scheduler(epoch):
            return lr * (decay ** (epoch // 10))
        x1,x2,y1,y2=train_test_split(self.imageData,self.labels,test_size=0.15,random_state=99)
        self.model=Sequential()
        self.model.add(Flatten(input_shape=(28,28,1)))
        self.model.add(Dense(128,activation='relu'))
        self.model.add(Dense(128,activation='relu'))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(26,activation='softmax'))
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        lr_schedule = LearningRateScheduler(lr_scheduler)
        self.model.fit(x1,y1,epochs=epochs,validation_data=(x2, y2),verbose=2)
        self.model.save('OCR.keras')
        

    def load_model(self,model_path):
        self.model=load_model(model_path)

    def diff_characters(self,path):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_array=np.array(img)
        characters=[]
        binary=np.where(img_array>127,1,0)
        i=0
        activity=[]
        for row in binary[:]:
            if row.sum()==0:
                activity.append(0)
            else:
                activity.append(1)
        lines=[]
        while i<len(activity)-1:
            while activity[i]!=1:
                if i>=len(activity)-1:
                    break
                i+=1
            y=i
            while activity[i]!=0:
                if i>=len(activity)-1:
                    break
                i+=1
            h=i
            if i<len(activity)-1:
                lines.append([y,h])
        linesimg=[]
        for row in lines:
            y,h=row[0],row[1]
            line=img_array[y:h]
            linesimg.append(np.array(line))
     
        for item in linesimg:
            activity=[]
            for row in item.T:
                #print(row)
                if row.sum()==0:
                    activity.append(0)
                else:
                    activity.append(1)
            #print(activity)
            blocks=[]
            i=0
            while i<len(activity)-1:
                gap=0
                while activity[i]!=1:
                    if i>=len(activity)-1:
                        break
                    gap+=1
                    i+=1
                if gap>15:
                    blocks.append([-1,-1])
                y=i
                while activity[i]!=0:
                    if i>=len(activity)-1:
                        break
                    i+=1
                h=i
                if i<len(activity)-1:
                    blocks.append([y,h])
            for character in blocks:
                y,h=character[0],character[1]
                if y==-1 and h==-1:
                    characters.append(np.zeros((28,28)))
                    continue
                line=item[:,y:h]
                characters.append(np.array(line))
        
        return characters

    def predict(self,image_path):
        characters=self.diff_characters(image_path)
        predicted=''
        for ch in characters:
            new_image=np.array(ch)
            row,cols=new_image.shape
            new=np.zeros((28,28))
            offset_x=(28 - cols) // 2
            offset_y=(28 - row) // 2
            new[offset_y:offset_y+row, offset_x:offset_x+cols]=new_image
            new=new/255.0
            if not bool(np.any(new)):
                predicted+=' '
                continue
            new=new.reshape(28, 28, 1)
            new=np.array([new])
            prediction=self.model.predict(new,verbose=0)
            predicted_class=np.argmax(prediction)
            #print(prediction)
            predicted_label=self.lab.inverse_transform([predicted_class])[0]
            predicted+=predicted_label
        return predicted

class SentimentAnalysis:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.lis = []
        self.probab = []
        self.sets = set()
        self.sentiments = []
        self.sentimentslength = []
        self.wordsSentiment = []
        self.pSentiments = []
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_data(self):
        for i in self.data['line']:
            a = i.split(' ')
            for j in a:
                self.sets.add(j.upper())
        self.v = len(self.sets)

        for d in self.data['sentiment'].unique():
            self.sentiments.append(d)
            lis2 = []
            slen = 0
            for i in self.data[self.data['sentiment'] == d]['line']:
                words = i.split(" ")
                for word in words:
                    lis2.append(word)
                slen += 1
            self.lis.append(lis2)
            self.sentimentslength.append(slen)

        for item in self.lis:
            self.wordsSentiment.append(len(item))

        self.sentimentslength = np.array(self.sentimentslength)
        totalLen = self.sentimentslength.sum()

        for i in self.sentimentslength:
            self.pSentiments.append(i / totalLen)

    def predict(self, text):
        string = text
        inputArray = string.split(" ")
        processed_input = []

        for word in inputArray:
            stemmed = self.stemmer.stem(word)
            lemmatized = self.lemmatizer.lemmatize(stemmed.upper())
            processed_input.append(lemmatized.upper())

        self.probab=[]
        for k in processed_input:
            templis = []
            for l, num in zip(self.lis, self.wordsSentiment):
                templis.append((l.count(k) + 1) / (num + self.v))
            self.probab.append(templis)
        self.probab = np.array(self.probab).prod(axis=0) * self.pSentiments
        return self.sentiments[np.argmax(self.probab)]



ocr=ocr_rec(r"D:\Code\Datasets\datasets-20240624T101648Z-001\datasets\alphabets_dataset\alphabet_labels.csv",r"D:\Code\Datasets\datasets-20240624T101648Z-001\datasets\alphabets_dataset\alphabet_images")
sentiment_analysis=SentimentAnalysis(r"D:\Code\Datasets\datasets-20240624T101648Z-001\datasets\sentiment_analysis_dataset.csv")
sentiment_analysis.preprocess_data()
#ocr.load_data('imgpixels.npy')
#ocr.train_model(epochs=12)
ocr.load_model('OCR.keras')
for i in range(1,7):
    predicted_label=ocr.predict(rf"D:\Code\Datasets\target_images-20240628T144514Z-001\target_images\line_{i}.png")
    print(f"Predicted label:{str(predicted_label)}")

    text=str(predicted_label)
    predicted_sentiment=sentiment_analysis.predict(text)
    print(f"Predicted sentiment:{predicted_sentiment}")
