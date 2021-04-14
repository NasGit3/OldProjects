import csv
import numpy as np
import cv2
import os
import pandas as pd

projectFolder = os.getcwd()

image = cv2.imread(projectFolder + '/mnist/train/0/00001.png',0)
imageReshaped =  np.reshape(image,(1,image.size))
print np.shape(imageReshaped)


columns=[]
for i in range(0,image.size):
    columns.append('Pixel'+str(i))

with open(projectFolder+'/testData.csv', 'w') as f1:
      w = csv.writer(f1, delimiter=',', lineterminator='\n')
      imageReshaped =np.append(imageReshaped,[5])
      listToWrite = imageReshaped.tolist()

      print type(listToWrite)
      # listToWrite.append(5)
      print listToWrite
      # w.writerow(columns)
      w.writerow(listToWrite)
# a = np.asarray([imageReshaped])
# np.savetxt("testDatacsv", a, delimiter=",", fmt='%1.4e')

df = pd.read_csv("testData.csv", names= columns)
# df = df.astype(str)
# print("Number of samples: %d" % len(df))
# df.head()


df = df.convert_objects(convert_numeric=True)


    # astype
df= df.astype('int')
print df
df.to_csv('testData2.csv')