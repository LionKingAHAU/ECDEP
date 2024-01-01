"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-0607
In the final part, we utilize the sub-cellular localization and communities as the feature of proteins.
Additionally, we use a sample method to alleviate the unbalanced learning problem.
"""

import numpy as np
import tensorflow as tf
from tqdm.notebook import trange
from tensorflow.keras import layers, Model, optimizers, losses, metrics, Input
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score


# Selected Community Process
def _dense_bn_com(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        bn = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(units=units_2, activation='relu')(bn)
        return dense2

    return f


# Sub-cellular Localization Process
def _dense_bn_sub(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]
    units_3 = params["units_3"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        dense2 = layers.Dense(units=units_2, activation='relu')(dense1)
        bn = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(units=units_3, activation='relu')(bn)
        return dense3

    return f


class ECDEP(object):
    @staticmethod
    def build(input_shape_sub, input_shape_com):
        # sub-cellular  part
        input_sub = Input(shape=input_shape_sub, name='subloc')
        output_sub = _dense_bn_sub(units_1=128, units_2=64, units_3=16)(input_sub)

        # Selected Community part
        input_com = Input(shape=input_shape_com, name='com')
        output_com = _dense_bn_com(units_1=64, units_2=16)(input_com)

        concat = layers.Concatenate(axis=-1)([output_sub, output_com])
        output = layers.Dense(units=1, activation='sigmoid')(concat)

        model = Model(inputs={'subloc': input_sub, 'com': input_com}, outputs=output)
        return model


class TrainData(object):
    def __init__(self, dataPath):
        subloc = np.load(dataPath + 'subcellular.npy')
        subloc = np.concatenate((subloc[:, 0:11], subloc[:, 65:1024]), axis=1)
        com = np.load(dataPath + 'community_selected.npy')
        label = np.load(dataPath + 'label.npy')

        # 2. first shuffle
        self.num = len(label)
        label = label.reshape((self.num, 1))
        shuffle_index = np.random.permutation(self.num)
        subloc = subloc[shuffle_index]
        com = com[shuffle_index]
        label = label[shuffle_index]

        # 3. split train set
        self.trainSub = subloc[:int(train_eval_rate * self.num)]
        self.trainCom = com[:int(train_eval_rate * self.num)]
        self.trainLabel = label[:int(train_eval_rate * self.num)]

        # 4. print information
        print("training data numbers(%d%%): %d" % (train_eval_rate * 100, len(self.trainLabel)))
        # 5. strip the pos and neg index
        self.pos_idx = (self.trainLabel == 1).reshape(-1)
        self.neg_idx = (self.trainLabel == 0).reshape(-1)

        # 6. get the size of train set and print the num of negative and positive amount in the train set
        self.training_size = len(self.trainLabel[self.pos_idx]) * 2
        print("positive data numbers", str(self.training_size // 2))
        print("negative data numbers", str(len(self.neg_idx)))

        # 7. split the test set
        self.test_S = subloc[int(train_eval_rate * self.num):]
        self.test_C = com[int( train_eval_rate * self.num):]
        self.test_Y = label[int(train_eval_rate * self.num):]
        self.test_size = len(self.test_Y)

    def shuffle(self):
        # 1. shuffle the negative part
        mark = list(range(int(np.sum(self.neg_idx))))
        np.random.shuffle(mark)

        # 2. even the neg and pos num in the train set
        self.train_C = np.concatenate(
            [self.trainCom[self.pos_idx], self.trainCom[self.neg_idx][mark][:self.training_size // 2]])
        self.train_S = np.concatenate(
            [self.trainSub[self.pos_idx], self.trainSub[self.neg_idx][mark][:self.training_size // 2]])
        self.train_Y = np.concatenate(
            [self.trainLabel[self.pos_idx], self.trainLabel[self.neg_idx][mark][:self.training_size // 2]])

        # 3. shuffle the train set concatenated above
        mark = list(range(self.training_size))
        np.random.shuffle(mark)
        self.train_C = self.train_C[mark]
        self.train_S = self.train_S[mark]
        self.train_Y = self.train_Y[mark]


def test(dataset, label):
    pred = model(dataset, training=False)

    acc = metrics.BinaryAccuracy()(label, pred)
    pre = metrics.Precision()(label, pred)
    rec = metrics.Recall()(label, pred)
    auc = metrics.AUC()(label, pred)
    ap = average_precision_score(label, pred)

    fpr, tpr, t = roc_curve(label, pred)
    precision, recall, tr = precision_recall_curve(label, pred)

    ypred = tf.math.greater(pred, tf.constant(0.5))
    ypred = tf.keras.backend.eval(ypred)

    tn, fp, fn, tp = confusion_matrix(label, ypred).ravel()
    F1 = f1_score(label, ypred)
    Spe = tn / (tn + fp)
    NPV = tn / (tn + fn)

    print("--------Start to TEST--------")
    print('- Accuracy %.4f' % acc)
    print('- Precision %.4f' % pre)
    print('- Recall %.4f' % rec)
    print('- F1-score %.4f' % F1)
    print('- Specificity %.4f' % Spe)
    print('- NPV %.4f' % NPV)
    print('- AUC %.4f' % auc)
    print('- AP %.4f' % ap)



if __name__ == '__main__':

    train_eval_rate = 0.8
    path = "../data/Dynamic Network Demo/Input Data/"
    epochs = 20
    batch_size = 64
    tLoss = []
    tAcc = []

    data = TrainData(path)
    model = ECDEP.build(input_shape_com=[64, ], input_shape_sub=[970, ])

    # define loss function and optimizer function
    loss_fun = losses.BinaryCrossentropy(from_logits=False)
    opt_fun = optimizers.Adamax(learning_rate=0.001)
    train_loss = metrics.Mean()
    train_acc = metrics.BinaryAccuracy()


    @tf.function
    def train_fun(dataset, label):
        with tf.GradientTape() as tape:
            pred = model(dataset, training=True)
            loss = loss_fun(label, pred)
        gradient = tape.gradient(loss, model.trainable_variables)
        opt_fun.apply_gradients(zip(gradient, model.trainable_variables))

        train_loss(loss)
        train_acc(label, pred)


for ep in trange(epochs):
    data.shuffle()
    for iter, idx in enumerate(range(0, data.training_size, batch_size)):
        # zip the data together as input
        batch_C = data.train_C[idx:idx + batch_size]
        batch_S = data.train_S[idx:idx + batch_size]
        batch_dict = {'subloc': batch_S, 'com': batch_C}
        batch_Y = data.train_Y[idx:idx + batch_size]

        # Reset the states of loss and acc function every epoch
        train_loss.reset_states()
        train_acc.reset_states()

        # train and validate the data, and compute the loss、acc, optimize the weights in the model network
        train_fun(batch_dict, batch_Y)

        if iter % 10 == 0:
            # record the loss and acc value to draw the curve
            tLoss.append(train_loss.result())
            tAcc.append(train_acc.result())
            print("=====epoch:%d iter:%d=====" % (ep + 1, iter + 1))
            print('- loss: %.4f' % train_loss.result())
            print('- binary_accuracy %.4f' % train_acc.result())
test_dict = {'subloc': data.test_S, 'com': data.test_C}
test(test_dict, data.test_Y)