import os
# go through all of the files and retrive the name of files
def file_name(file_dir):
    L = []
    for root, dirs, files, in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))
    return L

fn = file_name("/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/NKI_Training")
