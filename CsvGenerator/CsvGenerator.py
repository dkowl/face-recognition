MAIN_DIR = r'C:\Users\Dominik\Desktop\ISEL\CV\ComputerVision\data\ATnT''\\'
TRAINING_DIR = r'training''\\'
TEST_DIR = r'test''\\'
DATA_DIR = r'..''\\'
FOLDER_PREFIX = 's'
FOLDER_NO = 40
PICTURE_NO = 10
TRAINING_PICS_PER_FOLDER = 9
FILE_SUFFIX = '.pgm'
FILENAME = 'classes.csv'

fTraining = open(MAIN_DIR + TRAINING_DIR + FILENAME, 'w+')
fTest = open(MAIN_DIR + TEST_DIR + FILENAME, 'w+')

for i in range(FOLDER_NO):
    for j in range(PICTURE_NO):
        
        line = DATA_DIR + FOLDER_PREFIX + str(i+1) + "\\" + str(j+1) + FILE_SUFFIX + ';' + str(i+1) + '\n'
        if j < TRAINING_PICS_PER_FOLDER:
            fTraining.write(line)
        else:
            fTest.write(line)





