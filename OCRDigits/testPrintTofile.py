import os
import ntpath

# Producing a list of all the distorted data in a text file. This is used for DIGITS frame Work
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

projectFolder = os.getcwd()

f1=open('./distortedDataLabels', 'w+')
listOfFolders = [os.path.join(root, name)
                 for root, dirs, files in os.walk(projectFolder)
                 for name in dirs]
print listOfFolders
for dir in listOfFolders:
    print dir
    if 'distorted' in dir:
        listOfFiles = [os.path.join(root, name)
                       for root, dirs, files in os.walk(dir)
                       for name in files]
        for file in listOfFiles:

            label = path_leaf(dir)
            f1.write(file + ' ' + label[0] +'\n')

f1.close()
