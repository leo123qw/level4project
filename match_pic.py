# -*- coding: utf-8 -*-
import time
# go through all of the files and retrive the name of files
import os
import shutil # python file copy
def readfilename(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os. path.join(path ,filename)
        if os.path.isdir(filepath):
            readfilename(filepath, allfile)
        else:
            allfile.append(filepath)

    return allfile

if __name__ =='__main__':
    path1= '/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/NKI_Training'
    path2 = '/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/NKI_Training'
    path3 = '/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/NKI_Training'
    allfile1 = []
    allfile1 = readfilename(path1,allfile1)
    allname1 = []
    allname2 = []
    startwith_0 = []
    for name1 in allfile1:
        #print name1
        t1 = name1.split(".")[0].split("/")[-1].split("_")
        print t1
        allname1.append(t1)

        if t1[0] == "0":
            for name2 in allfile1:
                t2 =name2.split(".")[0].split("/")[-1].split("_")
                if t1[1:] == t2[0:]:
                    t1=t2
                    print t1,t2

    print("----------------")
