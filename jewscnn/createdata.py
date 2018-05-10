import os

import numpy as np
from PIL import Image
from pandas.core.frame import DataFrame

imgdir = os.listdir("jews")
data = []
for eachdir in imgdir:
    lable = int(eachdir[0])#*3 + int(eachdir[1])
    for eachimg in os.listdir("jews\\"+eachdir):
        datarow = [lable]
        img = Image.open("jews\\"+eachdir+"\\"+eachimg)
        img = img.crop((20, 20, 44, 44))
        imgdata = np.array(list(img.getdata())).T
        imgdata = imgdata.reshape(-1, )
        datarow.extend(imgdata.tolist())
        data.append(datarow)

pddata = DataFrame(data)
pddata.rename(columns={0: 'label'}, inplace=True)
pddata.to_csv("train.csv", index=False)
# np.savetxt("train.csv", np.array(data).astype(np.int8), delimiter=",")
# print(pddata)
