import tensorflow as tf
import pandas as pd
import numpy as np
from random import sample

logdir = ".\\tflog"
# Load data
train_data = pd.read_table("train.csv", sep=',')
test_data = pd.read_table("test.csv", sep=',')


# Split train and test
# train_index = np.random.randint(0,data.shape[0]-1,data.shape[0]/2-1)
train = train_data.drop('label', 1)
Y = train_data['label']

test = test_data.drop('label', 1)
Y_test = test_data['label']

# Reshape data
images = train.iloc[:, :].values
images = images.astype(np.float)

test_images = test.iloc[:, :].values
test_images = test_images.astype(np.float)

# Convert from [0:255] to [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
images = images.reshape(-1, 3, 576)

test_images = np.multiply(test_images, 1.0 / 255.0)
test_images = test_images.reshape(-1, 3, 576)

# New train data recognized by human , bind into label and image
# new_label = pd.read_table('~/Github/Python/tf/obscure_digit.csv',sep=',')
# new_train = pd.read_table('~/Github/Python/tf/obscure_images.csv',sep=',')

# new_train = new_train.iloc[:,:].values
# images = np.concatenate((new_train, images), axis = 0)

# Y = pd.concat((new_label.ix[:,0], Y), axis=0)

# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels_flat = Y.values.ravel()
labels_test_flat = Y_test.values.ravel()
# labels_count = 22
labels_count = 3

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

labels_test = dense_to_one_hot(labels_test_flat, labels_count)
labels_test = labels_test.astype(np.uint8)

'''
# Load test data
test = pd.read_table("test.csv", sep=',')

images_test = test.iloc[:, :].values
images_test = images_test.astype(np.float)
images_test = np.multiply(images_test, 1.0 / 255.0)
images_test = images_test.reshape(-1, 3, 576)
'''
# Set parameters
batch_size = 16
train_iters = 3000
display_step = 10
save_step = 200

# Dynamically adjust learning rate
global_step = tf.Variable(0, trainable=False)
# Stuck when starting learning rate is 0.1, 0.05, 0.01
learning_rate = tf.train.exponential_decay(
    0.001,  # Base learning rate.
    global_step,  # Current index into the dataset.
    train_iters // 15,  # Decay step.
    0.9,  # Decay rate.
    staircase=True)

# Network Parameters
n_input = 24*24
n_depth = 3
n_class = 3
# No drop out
dropout = 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_depth, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_class], name="y")

# Drop (keep probability)
keep_prob = tf.placeholder(tf.float32, name="keep_prob")


# Creat model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1],
                                                  padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Store layers weight & bias
# 5x5 conv, 1 input, 32 outputs
wc1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.5), name="wc1")
bc1 = tf.Variable(tf.zeros([32]), name="bc1")

# 5x5 conv, 32 inputs, 64 outputs
wc2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="wc2")
bc2 = tf.Variable(tf.random_normal(shape=[64]), name="bc2")

# 5x5 conv, 64 inputs, 128 outputs
wc3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1), name="wc3")
bc3 = tf.Variable(tf.random_normal(shape=[128]), name="bc3")

# 5x5 conv, 128 inputs, 256 outputs
wc4 = tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=0.1), name="wc4")
bc4 = tf.Variable(tf.random_normal(shape=[256]), name="bc4")

# Fully connected, 256 inputs, 1024 outputs
wd1 = tf.Variable(tf.truncated_normal([1 * 1 * 256, 1024], stddev=0.1), name="wd1")
bd1 = tf.Variable(tf.random_normal(shape=[1024]), name="bd1")

# 1024 inputs, 10 ouputs (class prediciton)
wout = tf.Variable(tf.truncated_normal([1024, n_class], stddev=0.1), name="wout")
bout = tf.Variable(tf.random_normal(shape=[n_class]), name="bout")

# Construct model
_X = tf.reshape(x, shape=[-1, 24, 24, 3])

# (Convolution Layer, MaxPooling, Dropout)*4
conv1 = conv2d(_X, wc1, bc1)
conv1 = max_pool(conv1, k=3)
conv1 = tf.nn.dropout(conv1, keep_prob)

conv2 = conv2d(conv1, wc2, bc2)
conv2 = max_pool(conv2, k=3)
conv2 = tf.nn.dropout(conv2, keep_prob)

conv3 = conv2d(conv2, wc3, bc3)
conv3 = max_pool(conv3, k=4)
conv3 = tf.nn.dropout(conv3, keep_prob)

conv4 = conv2d(conv3, wc4, bc4)
conv4 = max_pool(conv4, k=2)
conv4 = tf.nn.dropout(conv4, keep_prob)

# Full connected layer, Relu activation, Apply dropout
# Reshape conv4 output to fit dense layer input
dense1 = tf.reshape(conv4, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
dense1 = tf.nn.dropout(dense1, keep_prob)

# Output, class prediction
pred = tf.nn.softmax(tf.matmul(dense1, wout) + bout, name="prediction")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(dense1, wout) + bout, labels=y))
tf.summary.scalar('loss', cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(cost, global_step=global_step)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()
modelsaver = tf.train.Saver(max_to_keep=1)
# Launch the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)  # 将训练日志写入到logs文件夹下
    sess.run(init)
    for i in range(train_iters):
        idx = sample(range(4048), batch_size)
        batch_xs = images[idx]
        batch_ys = labels[idx]
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if i % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter" + str(i) + ", Minibatch Loss= " + "{:.15f}".format(loss))
            rs = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            writer.add_summary(rs, i)

        if i % save_step == 0:
            test_count = 0
            true_predict_count = 0
            test_count_map = dict()
            true_predict_count_map = dict()

            for j in range(len(test_images)//batch_size):
                idx = [x+j*batch_size for x in range(batch_size)]
                batch_test_x = test_images[idx]
                batch_test_y = labels_test[idx]

                predict_res = sess.run(pred, feed_dict={x: batch_test_x, keep_prob: dropout})

                batch_test_label = np.argmax(labels_test[idx], 1)
                predict_label = np.argmax(predict_res, 1)
                for ii in range(batch_size):
                    test_count += 1
                    if batch_test_label[ii] not in test_count_map:
                        test_count_map[batch_test_label[ii]] = 0
                    test_count_map[batch_test_label[ii]] += 1
                    if batch_test_label[ii] == predict_label[ii]:
                        true_predict_count += 1
                        if batch_test_label[ii] not in true_predict_count_map:
                            true_predict_count_map[batch_test_label[ii]] = 0
                        true_predict_count_map[batch_test_label[ii]] += 1

                # print('batch_test_y:', np.argmax(batch_test_y, 1))
                # print('predict_res:', predict_res)
                # print('predict_res:', np.argmax(predict_res, 1))
            modelsaver.save(sess, "..\\..\\model\\level\\levelmodel.ckpt", global_step=global_step)

            print('\nall:{}, true:{}, rate:{} \n all_map:{}, true_map:{}\n'.format(test_count, true_predict_count, true_predict_count/test_count, test_count_map, true_predict_count_map))

    print("train Finished!")

writer.close()

