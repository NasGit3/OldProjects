import sys
import os
class cd:


    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

projectPath = os.getcwd()
trainDataFolderPath = projectPath + '/mnist/train/'
print trainDataFolderPath
#  this line checks if the folder exist
if not(os.path.isdir(trainDataFolderPath)):
    print trainDataFolderPath + ' is not a directory'
    sys.exit()



#  To get the folder list that satisfy a condition
listOfFolders = [os.path.join(root,name)
                    for root, dirs, files in os.walk(trainDataFolderPath)
                    for name in dirs]
print listOfFolders
listOfFolders.sort()
print listOfFolders

# for dir in listOfFolders:
#     with cd(dir):
#          # do something


