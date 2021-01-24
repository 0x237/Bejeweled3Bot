import os

import numpy as np
from PIL import Image
from pandas.core.frame import DataFrame

imgdir = os.listdir("..\\jews")
train_data = list()
test_data = list()
for eachdir in imgdir:
    cnt = 0
    lable = int(eachdir[1])
    for eachimg in os.listdir("..\\jews\\"+eachdir):
        datarow = [lable]
        img = Image.open("..\\jews\\"+eachdir+"\\"+eachimg)
        img = img.resize((24, 24))
        imgdata = np.array(list(img.getdata())).T
        imgdata = imgdata.reshape(-1, )
        datarow.extend(imgdata.tolist())
        if cnt < 16:
            test_data.append(datarow)
        else:
            train_data.append(datarow)

        cnt += 1

train_pddata = DataFrame(train_data)
train_pddata.rename(columns={0: 'label'}, inplace=True)
train_pddata.to_csv("train.csv", index=False)

test_pddata = DataFrame(test_data)
test_pddata.rename(columns={0: 'label'}, inplace=True)
test_pddata.to_csv("test.csv", index=False)
