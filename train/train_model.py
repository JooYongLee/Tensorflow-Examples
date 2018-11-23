# from model import NerveTrackNet
import model
import tensorflow as tf
import cv2
# from tensorflow.contrib import layers
import cnfig
import numpy as np
import cifar10_reader

def load_cifar_image_set():
    image_class_names = cifar10_reader.load_class_names()
    Images, cls, one_hot_encoded= cifar10_reader.load_training_data()
    return image_class_names, {
        "images":Images,
        "labels":cls,
        "one_hot_labels":one_hot_encoded
    }

def reshape_img(imgs):
    rsz_img = []
    rsz_shape = (cnfig.IMAGE_WIDTH, cnfig.IMAGE_HEIGHT)
    for id in range(imgs.shape[0]):
        rsz_img.append(cv2.resize(imgs[id],rsz_shape))
    return np.array(rsz_img)

def get_random_set(image, labels):
    total_sample_size = image.shape[0]
    rand_mask = np.random.choice(total_sample_size, cnfig.BATCH_SIZE)
    sample_img = image[rand_mask]
    return reshape_img(sample_img), labels[rand_mask]

def nerve_train():
    """
    :return:
    """
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, cnfig.IMAGE_HEIGHT, cnfig.IMAGE_WIDTH, 3],
                            name='nerve_track_inputs')
    labels = tf.placeholder(dtype=tf.float32, shape=[None, cnfig.NUM_CALSS], name='labels')

    class_name_list, dataset = load_cifar_image_set()
    imgset = dataset["images"]
    labelsset = dataset["labels"]
    labels_one_hot_set = dataset["one_hot_labels"]

    print("-----------------")
    print("------imgset-------", imgset.shape)
    print("-------labels_one_hot_set---", labels_one_hot_set.shape)

    # ind = np.random.choice(10000)
    # cv2.imshow(class_name_list[labelsset[ind]], imgset[ind])
    # cv2.waitKey(0)

    nerve_models = model.NerveTrackNet(inputs)


    loss = tf.nn.softmax_cross_entropy_with_logits(logits=nerve_models.fc6, labels=labels)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(cnfig.LEARNING_RATE).minimize(loss)

    data_set_size = np.shape(imgset)[0]



    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    iteration = int(data_set_size / cnfig.BATCH_SIZE) + 1
    for ep in range(cnfig.TRAIN.EPOCH):
        batch_mask = np.random.permutation(data_set_size)
        extractor = np.linspace(0,data_set_size,iteration+1, dtype=np.int)
        for it in range(iteration):
            start = extractor[it]
            end = extractor[it+1]

            batch_img = imgset[start:end]
            batch_img = reshape_img(batch_img)
            batch_lab = labels_one_hot_set[start:end]

            feeds = {
                inputs : batch_img,
                labels : batch_lab
            }
            _, calc_loss = sess.run([optimizer, loss], feeds)

            if it % 4000:
                test_accuryacy = 0
                test_cnts = 5
                for _ in range(test_cnts):
                    test_img, test_labels = get_random_set(imgset, labels_one_hot_set)

                    prediction = sess.run(nerve_models.softmax_fc6, feed_dict={inputs:test_img, labels:test_labels})
                    prediction_argmax = np.argmax(prediction,axis=1)
                    gts = np.argmax(batch_lab)
                    accuracy = np.sum(prediction_argmax == gts) / cnfig.BATCH_SIZE
                    test_accuryacy += accuracy
                test_accuryacy /=  test_cnts

                print("epoch{}=iter{}: loss {:.1f} , acuracy:{:.4f}".format(
                    ep, it, calc_loss, test_accuryacy
                ))





def cv_rsz_():
    x = np.zeros([100,100,3],dtype=np.uint8)
    y = cv2.resize(x,(200,200),)
    print(x.shape, y.shape)
if __name__ =="__main__":
    # cv_rsz_()
    nerve_train()
    pass

