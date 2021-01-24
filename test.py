
from PIL import Image
import tensorflow as tf
import numpy as np

levelgraph = tf.Graph()
with levelgraph.as_default():
    levelsaver = tf.train.import_meta_graph(".\\model\\level\\levelmodel.ckpt-601.meta")
    levelsess = tf.Session(graph=levelgraph)
    levelsaver.restore(levelsess, ".\\model\\level\\levelmodel.ckpt-601")

jewimg = Image.open(".\\tmp\\1611482077.8908546.jpg")
jewimg = jewimg.resize((24, 24))
imgdata = np.array(list(jewimg.getdata())).T
imgdata.reshape(-1, )
res = levelsess.run(levelgraph.get_tensor_by_name("prediction:0"),
                    feed_dict={levelgraph.get_tensor_by_name("x:0"): imgdata.reshape((-1, 3, 576)),
                                levelgraph.get_tensor_by_name("keep_prob:0"): 1})
label = np.argmax(res, 1)[0]
print(label)
