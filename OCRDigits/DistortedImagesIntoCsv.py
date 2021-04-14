import os
import ntpath
import numpy as np
import cv2
import csv
import pandas as pd


# Converting images to CSV file.

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Getting the project file
projectFolder = os.getcwd()

# Creating headers for csv file
columns=[]
columns.append('label')

for i in range(0,784):
    columns.append('pixel'+str(i))

# Parsing each sub-folder
listOfFolders = [os.path.join(root, name)
                 for root, dirs, files in os.walk(projectFolder)
                 for name in dirs
                 if not(os.path.join(root, name).__contains__('.git'))]   #if (name.startswith('train') | name.startswith('test'))
print listOfFolders

# Creating a CSV file
with open('./distortedData.csv', 'w') as f1:
    w= csv.writer(f1, delimiter=',', lineterminator='\n')

# Going through each sub-folder to get the images
    for dir in listOfFolders:
        print dir
        if 'distorted' in dir:
            listOfFiles = [os.path.join(root, name)
                       for root, dirs, files in os.walk(dir)
                       for name in files]
            for file in listOfFiles:
                image = 255-image

                # Converting each image to a row vector
                imageReshaped = np.reshape(image,(1,image.size))

                label = path_leaf(dir)
                imageReshaped = np.append(imageReshaped, label[0])

                # adding each reshaped image to a list
                listToWrite = imageReshaped.tolist()

                w.writerow(listToWrite)

df = pd.read_csv("distortedData.csv", names= columns)

# Converting the type of the data stored in csv to integer
df = df.convert_objects(convert_numeric=True)
df= df.astype('int')
# Shuffle the rows of the CSV file
df2=df.iloc[np.random.permutation(len(df))]
df2=df2.reset_index(drop=True)

df2.to_csv('distortedData.csv')


