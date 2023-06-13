from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import autoencoder_better.metrics as my_metrics
from autoencoder_better.ConvAE import CAE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import copyUtils.copyDirectoryUtils as copy_utils
from params import images_shape_x, images_shape_y, images_main_directory, get_file
import params as params
from sklearn.model_selection import train_test_split
import modelUtils.modelUtils as modelUtils
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])

        self.model.summary()
        plot_model(self.model,
                   to_file='C:/Users/wolukasz/Desktop/PG/II_stopien/Praca_Magisterska/REPORTS/2021-05-05/autoencoder/model_extended_plot.png',
                   show_shapes=True, show_layer_names=True)

    def pretrain(self, x, x_val, batch_size=256, epochs=500, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')

        # begin training
        t0 = time()

        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, validation_data=(x_val, x_val), callbacks=[
            EarlyStopping(monitor='val_loss', patience=300, verbose=0, mode='min'),
            ModelCheckpoint(params.autoencoder_weights_path, save_best_only=True, monitor='val_loss', mode='min')
        ])

        # self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, )
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def autoencoder_predict(self, x):
        _, q = self.model.predict(x, verbose=0)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, x_val=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=500')
            self.pretrain(x, x_val=x_val, batch_size=batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=500)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        #logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(my_metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(my_metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(my_metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y


def load_mnist_val():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x, y) = mnist.load_data()

    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y


def train_model(args, data, dcec):
    x = data["x"]
    y = data["y"]
    x_val = data["x_val"]
    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args["gamma"], 1], optimizer=optimizer)
    dcec.fit(x, y=y, x_val=x_val, tol=args["tol"], maxiter=args["maxiter"],
             update_interval=args["update_interval"],
             save_dir=args["save_dir"],
             cae_weights=args["cae_weights"])
    y_pred = dcec.y_pred
    print('acc = %.4f, nmi = %.4f, ari = %.4f' % (my_metrics.acc(y, y_pred), my_metrics.nmi(y, y_pred), my_metrics.ari(y, y_pred)))


def test_autoencoder(args, data, dcec):
    # load dataset
    x_val = data["x_val"]
    y_val = data["y_val"]

    # load weights
    #dcec.cae.load_weights('D:/dataset/autoencoderTest/pretrain_cae_model.h5')
    dcec.load_weights('D:/dataset/autoencoderTest/dcec_model_final.h5')

    # predict
    y = dcec.autoencoder_predict(x_val)

    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    for i in range(0, 3):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow((x_val[i]).reshape(images_shape_x, images_shape_y))
        fig.add_subplot(rows, columns, i + 1 + columns)
        plt.imshow((y[i]).reshape(images_shape_x, images_shape_y))
    plt.show()


def test_model(args, data, dcec):
    # load dataset
    x_val = data["x_val"]
    y_val = data["y_val"]
    val_images_array = data["val_images_array"]

    # load weights
    dcec.load_weights('D:/dataset/autoencoderTest/dcec_model_final.h5')

    # predict
    y = dcec.predict(x_val)

    print("[-1, 1] (Best 1) Adjusted Rand index: ", metrics.adjusted_rand_score(y_val, y))
    print("[-1, 1] (Best 1) Mutual Information based scores: ", metrics.adjusted_mutual_info_score(y_val, y))
    print("[-1, 1] (Best 1) V-measure: ", metrics.v_measure_score(y_val, y))

    copy_utils.save_predicted_clusters('D:/dataset/autoencoderTest/predicted', y, val_images_array)
    sns.set(font_scale=3)
    # print(y_val)
    # print(y)
    confusion_matrix = metrics.confusion_matrix(y_val, y)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.savefig(os.path.join(args["save_dir"], 'conf_matrix.png').__str__())


def load_images(paths):
    return np.array(list(map(lambda path: get_file(path, (images_shape_x, images_shape_y)), paths)))


if __name__ == "__main__":

    CAE((128, 128, 1), filters=[32, 64, 128, 256])
    DCEC(input_shape=(128, 128, 1), filters=[32, 64, 128, 256], n_clusters=15)
    # args = {
    #     "n_clusters": 15,
    #     "batch_size": 256,
    #     "maxiter": 20000,
    #     "gamma": 0.1,
    #     "update_interval": 140,
    #     "tol": 0.001,
    #     "cae_weights": 'D:/dataset/autoencoderTest/pretrain_cae_model.h5',
    #     "save_dir": 'D:/dataset/autoencoderTest'
    # }
    # onlyVal = True
    #
    # images_array, images_category_array = copy_utils.get_images_list(images_main_directory)
    # images_category_array = modelUtils.convert_to_numeric_labels(images_category_array)
    # train_images_array, val_images_array, y_train, y_val = train_test_split(images_array, images_category_array, test_size=0.2, random_state=42)
    #
    # X_train = []
    # if onlyVal is False:
    #     X_train = load_images(train_images_array)
    #
    # X_val = load_images(val_images_array)
    #
    #
    # #X_train, X_val, y_train, y_val, train_images_array, val_images_array = train_test_split(x, train_images_category_array, train_images_array, test_size=0.2, random_state=42)
    #
    # # x, y = load_mnist()
    # # x_val, y_val = load_mnist_val()
    # data = {
    #     "x": np.asarray(X_train),
    #     "train_images_array": np.asarray(train_images_array),
    #     "y": np.asarray(y_train),
    #     "x_val": np.asarray(X_val),
    #     "val_images_array": np.asarray(val_images_array),
    #     "y_val": np.asarray(y_val)
    # }
    #
    # # prepare the DCEC model  # [32, 64, 128, 10] <- 939,915 params <- stare [32, 64, 128, 256] <- 17,062,017 params
    # dcec = DCEC(input_shape=X_val.shape[1:], filters=[32, 64, 128, 256], n_clusters=args["n_clusters"]) #[32, 64, 128, 256] <- spróbować z takimi parametrami  https://arxiv.org/ftp/arxiv/papers/1710/1710.08961.pdf
    # # dcec.model.summary()
    #
    #
    # # X_val = X_val[0:10]
    # # pred = dcec.extract_feature(X_val)
    # # print(np.shape(pred))
    #
    # #train_model(args, data, dcec)
    # #test_autoencoder(args, data, dcec)
    # test_model(args, data, dcec)



# Epoch 362/2000
# 11400/11400 [==============================] - 48s 4ms/step - loss: 6.3890e-04 - val_loss: 0.0011

# Epoch 796/2000
# 11400/11400 [==============================] - 47s 4ms/step - loss: 4.8960e-04 - val_loss: 0.0011
# Pretraining time:  37658.558116436005

# Iter 15540 : Acc 0.23956 , nmi 0.27478 , ari 0.11761 ; loss= [0.01275 0.09174 0.00357]
# delta_label  0.0007894736842105263 < tol  0.001
# Reached tolerance threshold. Stopping training.
# saving model to: D:/dataset/autoencoderTest/dcec_model_final.h5
# Pretrain time:   37661.02592396736
# Clustering time: 16238.321730613708
# Total time:      53899.34765458107
# C:\Users\wolukasz\Anaconda3\envs\cokolwiek\lib\site-packages\sklearn\utils\linear_assignment_.py:128: FutureWarning: The linear_assignment function is deprecated in 0.21 and will be removed from 0.23. Use scipy.optimize.linear_sum_assignment instead.
#   FutureWarning)
# acc = 0.2396, nmi = 0.2748, ari = 0.1176
#
# Process finished with exit code 0
