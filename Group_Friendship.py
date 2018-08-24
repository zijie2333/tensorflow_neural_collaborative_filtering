'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
Friendship Struction Concat Friends Output.
'''

import numpy as np
import tensorflow as tf
import math
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import logging
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Dataset_New import Dataset
from time import time
import sys
import math
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Airline',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--Interaction_layers',default='[64,32,16]',
                        help='Embedding size of MF model.')
    parser.add_argument('--Divide_layers', nargs='?', default='[64,32,16]',
                        help="Divide layers. Divide each friend into a group, the first layer is twice the embedding size.")
    parser.add_argument('--l1', nargs='?', default=1e-8,
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--l2', nargs='?', default=1e-10,
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--l1l2', nargs='?', default=0,
                        help="Regularization for user and item em0eddings.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--num_items', type=int, default=3,
                        help='Number of stances.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes.')
    parser.add_argument('--num_users', type=int, default=11986,
                        help='Num of users.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='keep prob.')
    parser.add_argument('--pre_target', type=bool, default=False,
                        help='whether to precheck if a target is included in a tweet.')
    parser.add_argument('--log_path', type=str, default='LOGS/',
                        help='the folder of logging')
    parser.add_argument('--log_on', type=str,
                        default='[\'dataset\',\'pre_load\', \'learning_rate\',\'num_factors\',\'mlp_layers\',\'tmp_dim\',\'pre_target\',\'l1\',\'l2\',\'l1l2\']',
                        help='the parameter of fine-tunning')
    return parser.parse_args()

class get_model_NeuMF(object):
    """Friends_Model
    """
    def __init__(self, args, sess, friends_per_user):

        self.n_friends=args.num_users
        self.friends_per_user=friends_per_user
        self.learning_rate = args.learning_rate
        self.Divide_layers=eval(args.Divide_layers)   ##dim of h
        self.Interaction_layers= eval(args.Interaction_layers)  ##dim of h
        self.n_classes=3
        self.embedding_size=self.Divide_layers[0]/2
        self.scale1 = args.l1
        self.scale2 = args.l2
        self.dropout=args.dropout
        self.scale = args.l1l2
        self.activation=tf.nn.relu
        self.initializer = tf.random_normal_initializer(stddev=0.01)  ##initializer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  ##optimizer
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)  ##regularizer
        self.sess = sess


        self.build_input()
        self.build_var()
        self.pred = self.build_model()


        # Define loss and optimizer
        cost=tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.y, self.pred)) + self.scale * tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])


        var_list = [var for var in tf.trainable_variables()]
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads = tf.gradients(cost, var_list)
        train_op = opt.apply_gradients(zip(grads, var_list))
        init_op = tf.initialize_all_variables()

        self.cost = cost
        self.train_op = train_op
        self.sess.run(init_op)

    def build_input(self):
            self.user_features_input = tf.placeholder(tf.int32, [None, self.friends_per_user],name="user_features_input")
            self.item_input = tf.placeholder(tf.int32, [None, 1], name="item_input")
            self.y = tf.placeholder(tf.float32, [None, 1], name="y")
            self.user_weights= tf.placeholder(tf.float32,  [None,self.friends_per_user], name='User_weights')
            self.phase = tf.placeholder(tf.bool, name="phase")


    def build_var(self):
                with tf.variable_scope('Embedding'):
                    #Item and user embeddings
                    self.Item_Embeddings= tf.get_variable('Item_Embeddings',initializer=self.initializer([self.n_classes,self.embedding_size]))
                    self.User_Embeddings= tf.get_variable('User_Embeddings',initializer=self.initializer([self.n_friends,self.embedding_size]))


                with tf.variable_scope('Divide_Layers'):
                    #Divide_Layers
                    self.Divide_weights,self.Divide_bias=[],[]
                    for idx in xrange(0,len(self.Divide_layers)-1):
                        name = 'Divide_Layer%d' % idx
                        self.Divide_weights.append(tf.get_variable(name + "_weight", initializer=self.initializer(
                            [self.Divide_layers[idx], self.Divide_layers[idx+1]])))
                        self.Divide_bias.append(
                            tf.get_variable(name + "_bias", initializer=self.initializer([self.Divide_layers[idx+1]])))

                    self.Divide_prediction_weight=tf.get_variable("Divide_prediction" + "_weight", initializer=self.initializer(
                            [self.Divide_layers[-1], 3]))
                    self.Divide_prediction_bias = tf.get_variable("Divide_prediction" + "_bias",
                                                                    initializer=self.initializer([3]))

                with tf.variable_scope('Interaction_Layers'):
                    #Interaction Layers
                    self.Interaction_weights, self.Interaction_bias = [], []

                    # First layer is the concat layer for the three stance vectors.
                    self.Interaction_weights.append(tf.get_variable("Interaction_Layer0"+ "_weight", initializer=self.initializer(
                        [3*self.embedding_size, self.Interaction_layers[0]])))

                    self.Interaction_bias.append(tf.get_variable("Interaction_Layer0" + "_bias",
                                        initializer=self.initializer([self.Interaction_layers[0]])))

                    for idx in xrange(1, len(self.Interaction_layers)):
                        name = 'Interaction_Layer%d' % idx
                        self.Interaction_weights.append(tf.get_variable(name + "_weight", initializer=self.initializer(
                            [self.Interaction_layers[idx-1], self.Interaction_layers[idx]])))
                        self.Interaction_bias.append(
                            tf.get_variable(name + "_bias",
                                            initializer=self.initializer([self.Interaction_layers[idx]])))

                    self.Interaction_prediction_weight = tf.get_variable("Interaction_prediction" + "_weight",
                                                                    initializer=self.initializer(
                                                                        [self.Interaction_layers[-1], 1]))
                    self.Interaction_prediction_bias = tf.get_variable("Interatcion_prediction" + "_bias",
                                                                  initializer=self.initializer([1]))

    def build_model(self):
            with tf.device('/gpu:0'):
                    with tf.variable_scope('Embeddings'):
                        if self.phase == True:
                            Item_Vector = tf.nn.dropout(tf.nn.embedding_lookup(self.Item_Embeddings, self.item_input),self.dropout)
                            User_Vector = tf.nn.dropout(tf.nn.embedding_lookup(self.User_Embeddings, self.user_features_input),self.dropout)
                        else:
                            User_Vector = tf.nn.embedding_lookup(self.User_Embeddings, self.user_features_input)  #[b,n,e]
                            Item_Vector = tf.nn.embedding_lookup(self.Item_Embeddings, self.item_input)  # [b,1,e]

                    with tf.variable_scope('Divide_Layers'):

                        Concat_Vector = tf.concat(
                            [User_Vector, tf.tile(Item_Vector, tf.stack([1,self.friends_per_user,1]))], 2)  # [b,n,2e]
                        Output = tf.reshape(Concat_Vector, [-1, 2 * self.embedding_size])  # [b*n,2e]

                        for idx in xrange(0,len(self.Divide_layers)-1):
                            Output = tf.add(tf.matmul(Output, self.Divide_weights[idx]),
                                                          self.Divide_bias[idx]) # [b*n,3]

                        Output = tf.add(tf.matmul(Output, self.Divide_prediction_weight),
                                                          self.Divide_prediction_bias) # [b*n,3]
                        Output = tf.reshape(Output, [-1, self.friends_per_user, 3])  # [b,n,3]
                        Output = tf.split(Output, 3, 2)  # [b,n,1]
                        self.user_weights_expand = tf.expand_dims(self.user_weights, 2)  # [b,n,1]
                        Group_vectors = tf.concat(
                            [tf.reduce_sum((Output[0] * self.user_weights_expand) * User_Vector, 1),
                             tf.reduce_sum((Output[1] * self.user_weights_expand) * User_Vector, 1),
                             tf.reduce_sum((Output[2] * self.user_weights_expand) * User_Vector, 1)], 1)  # [b,3e]

                    with tf.variable_scope("Interaction_Layers"):
                            for idx in xrange(0,len(self.Interaction_layers)):
                                Group_vectors = tf.add(tf.matmul(Group_vectors,self.Interaction_weights[idx]),self.Interaction_bias[idx])
                            pred = tf.add(tf.matmul(Group_vectors,self.Interaction_prediction_weight),self.Interaction_prediction_bias) #[batch,1]

                    return pred


    def train_batch(self, u, i,y,w,phase):
     self.sess.run(self.train_op, feed_dict={self.user_features_input: u, self.item_input: i, self.y:y,self.user_weights:w,  self.phase: phase})

    def single_input_evaluate(self,user,label,w):
        preds=np.zeros(3,dtype="float")
        user=np.reshape(user,(1,-1))
        label=np.reshape(label,(-1,1))
        w=np.reshape(w,(1,-1))
        for x in xrange(0,3):
            u=user  #numpy
            i=np.zeros(1,dtype='int32')
            y=np.zeros(1,dtype='int32')
            i[0]=x
            if i[0] == label[0]:
                y[0]=1
            else:
                y[0]=0

            i=np.reshape(i,(-1,1))
            y=np.reshape(y,(-1,1))

            preds[x]= self.sess.run(self.pred, feed_dict={self.user_features_input: u, self.item_input: i, self.y: y, self.user_weights:w, self.phase: False})  # [1,1]

        predicted_label=np.where(preds==np.max(preds))
        prediction=predicted_label[0][0] #int

        return prediction

    def evaluate(self,u,y,w, isintarget,phase=False):
             batch_size=y.shape[0]
             y_pred=np.zeros(batch_size,dtype='int32')
             for x in xrange(0,y.shape[0]):
                 y_pred[x]=self.single_input_evaluate(u[x],y[x],w[x])

             y_true=y
             index_favors = np.argwhere( y_true == 2)
             index_againsts = np.argwhere( y_true == 1)
             index_nones = np.argwhere( y_true == 0)

             # Calculate seperate accuracy,F1_FAVOR,F1_AGAINST
             acc_none, acc_favor, acc_against = [], [], []
             TP_F, FN_F, FP_F, TP_A, FN_A, FP_A,TP_N = 0, 0, 0, 0, 0, 0, 0
             # ---favor
             for index_favor in index_favors:
                 index_favor = index_favor[0]
                 if y_pred[index_favor] == 2:
                     acc_favor.append(1)
                     TP_F += 1
                 else:
                     acc_favor.append(0)
                     FN_F += 1
                     if y_pred[index_favor] == 1:
                         FP_A += 1
             accuracy_favor = np.mean(acc_favor)

             # ---against
             for index_against in index_againsts:
                 index_against = index_against[0]
                 if y_pred[index_against] == 1:
                     acc_against.append(1)
                     TP_A += 1
                 else:
                     acc_against.append(0)
                     FN_A += 1
                     if y_pred[index_against] == 2:
                         FP_F += 1
             accuracy_against = np.mean(acc_against)

             # ---none
             for index_none in index_nones:
                 index_none = index_none[0]
                 if y_pred[index_none] == 0:
                     acc_none.append(1)
                     TP_N+=1
                 else:
                     acc_none.append(0)
                     if y_pred[index_none] == 1:
                         FP_A += 1
                     if y_pred[index_none] == 2:
                         FP_F += 1
             accuracy_none = np.mean(acc_none)

             F1_FAVOR = 2 * float(TP_F) / float(2 * TP_F + FP_F + FN_F)
             F1_AGAINST = 2 * float(TP_A) / float(2 * TP_A + FP_A + FN_A)

             accuracy_whole = float(TP_N+TP_F+TP_A)/y_true.shape[0]

             return accuracy_whole, 'DK', accuracy_none, accuracy_against, accuracy_favor, F1_FAVOR, F1_AGAINST

    def save(self,out_file):
        saver=tf.train.Saver()
        save_path = saver.save(self.sess, out_file)
        print("saved!")

def get_batch(x,f,y,w, step, batch_size):
    start = step * batch_size
    total= x.shape[0]

    batch_x = []
    batch_f = []
    batch_y = []
    batch_w = []
    if start+batch_size<=total:
      for i in range(batch_size):
        batch_y.append(y[(i + start)])
        batch_f.append(f[(i + start)])
        batch_x.append(x[(i + start)])
        batch_w.append(w[(i + start)])
    else:
      length=total-start
      for i in range(0,length-1):
          batch_y.append(y[(i + start)])
          batch_f.append(f[(i + start)])
          batch_x.append(x[(i + start)])
          batch_w.append(w[(i + start)])

    return np.asarray(batch_x), np.asarray(batch_f), np.asarray(batch_y),np.asarray(batch_w)

def random_shuffle(a, b, c,d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)


if __name__ == '__main__':
    np.random.seed(1337)
    tf.set_random_seed(1337)

    args = parse_args()
    args_str = str(args)
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    print("MLP arguments: %s " % (args))

    save_path = 'Pretrain/%s/NeuMF/' % args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_file = save_path + '%s_NeuMF_%d.h5' % (args.dataset, time())

    # Writing Logs.
    log_name = "".join([args_str[args_str.find(str + "="):].split(",")[0] for str in eval(args.log_on)])
    args.log_path=args.log_path+args.dataset+"/"
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M %S',
                        filename=os.path.join(args.log_path, "Neuro_%s.txt" % log_name))
    print(args)
    logging.info("NeuMF_%s!============================================" % args.dataset)
    logging.info(args)



    # Loading data
    t1 = time()
    dataset = Dataset(args)
    train_users, train_items, train_y,train_weights = dataset.train_users, dataset.train_items, dataset.train_labels, dataset.train_weights
    test_users,test_y, test_istarget,test_weights = dataset.test_users, dataset.test_labels, dataset.test_isintargets,dataset.test_weights
    t_train_users, t_train_y, train_istarget,t_train_weights= dataset.t_train_users, dataset.t_train_labels, dataset.t_train_isintargets,dataset.t_train_weights
    friends_per_user=dataset.largest_friends_number

    num_users, num_items = args.num_users, args.num_items
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train_users.shape[0], test_users.shape[0]))

    # Build model
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = get_model_NeuMF(args,sess,friends_per_user)

    # Prepare for figures.
    train_accuracy_display = np.zeros((4, args.epochs+1))
    train_F1 = np.zeros((3, args.epochs+1))
    test_accuracy_display = np.zeros((4, args.epochs+1))
    test_F1 = np.zeros((3, args.epochs+1))


    # Init performance
    t1 = time()
    (Train_Acc, Train_Loss, acc_N_Train, acc_A_Train, acc_F_Train, F_1_favor_Train, F_1_against_Train) = model.evaluate(
        t_train_users, t_train_y,t_train_weights,train_istarget, False)
    (Test_Acc, Test_Loss, acc_N_Test, acc_A_Test, acc_F_Test, F_1_favor_Test, F_1_against_Test) = model.evaluate(
        test_users, test_y, test_weights,test_istarget, False)

    # For Drawing
    train_accuracy_display[(0, 0)] = Train_Acc
    train_accuracy_display[(1, 0)] = acc_N_Train
    train_accuracy_display[(2, 0)] = acc_A_Train
    train_accuracy_display[(3, 0)] = acc_F_Train
    train_F1[(0, 0)] = F_1_favor_Train
    train_F1[(1, 0)] = F_1_against_Train
    train_F1[(2, 0)] = (F_1_favor_Train + F_1_against_Train) * 0.5

    test_accuracy_display[(0, 0)] = Test_Acc
    test_accuracy_display[(1, 0)] = acc_N_Test
    test_accuracy_display[(2, 0)] = acc_A_Test
    test_accuracy_display[(3, 0)] = acc_F_Test
    test_F1[(0, 0)] = F_1_favor_Test
    test_F1[(1, 0)] = F_1_against_Test
    test_F1[(2, 0)] = (F_1_favor_Test + F_1_against_Test) * 0.5

    str1="Init: Train_F1_avg = %.4f, Test_F1_avg %.4f, Train_Acc = %.4f, Test_Acc = %.4f,  Train_Loss = %s, Test_Loss = %s  [%.1f]" % ((F_1_favor_Train + F_1_against_Train) * 0.5, (F_1_favor_Test + F_1_against_Test) * 0.5, Train_Acc, Test_Acc,Train_Loss, Test_Loss, time() - t1)
    str2="Init:Train----------------------\r\n \t\t   Accuracy \t F_1 Score \r\n Against \t %.4f  \t%.4f  \r\n Favor   \t %.4f  \t%.4f   \r\n None   \t %.4f\r\n" % (acc_A_Train, F_1_against_Train, acc_F_Train, F_1_favor_Train, acc_N_Train)
    str3="Init:Test----------------------\r\n \t\t   Accuracy \t F_1 Score \r\n Against \t %.4f  \t%.4f  \r\n Favor   \t %.4f  \t%.4f   \r\n None   \t %.4f\r\n" % (acc_A_Test, F_1_against_Test, acc_F_Test, F_1_favor_Test, acc_N_Test)
    strs=np.array([str1,str2,str3])
    for string in strs:
        print(string)
        logging.info(string)



    # Train model
    best_acc, best_loss, best_F1_avg, best_F1_against, best_F1_favor, best_epoch = Test_Acc, Test_Loss, (F_1_against_Test + F_1_favor_Test) * 0.5, F_1_against_Test, F_1_favor_Test, -1
    best_accuracy = best_acc
    best_epoch_accuracy = best_epoch
    total_batch = int(math.ceil(train_users.shape[0] / float(batch_size)))
    for epoch in xrange(epochs):
        t1 = time()
        random_shuffle(train_users, train_items, train_y, train_weights)
        for i in xrange(total_batch):
            batch_u, batch_i, batch_y,batch_w = get_batch(train_users, train_items, train_y,train_weights, i, batch_size=batch_size)
            model.train_batch(batch_u, batch_i, batch_y, batch_w,True)

        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (Train_Acc, Train_Loss, acc_N_Train, acc_A_Train, acc_F_Train, F_1_favor_Train,F_1_against_Train) = model.evaluate(t_train_users, t_train_y, t_train_weights,train_istarget, False)
            (Test_Acc, Test_Loss, acc_N_Test, acc_A_Test, acc_F_Test, F_1_favor_Test, F_1_against_Test) = model.evaluate(test_users, test_y, test_weights, test_istarget, False)
            str1='Epoch %d [%.1f s]:Train_F1_avg = %.4f, Test_F1_avg %.4f,  Train_Acc = %.4f, Test_Acc = %.4f,  Train_Loss = %s, Test_Loss = %s [%.1f s]'% (epoch, t2 - t1, (F_1_favor_Train + F_1_against_Train) * 0.5,(F_1_favor_Test + F_1_against_Test) * 0.5, Train_Acc, Test_Acc, Train_Loss, Test_Loss,time() - t2)
            str2="Epoch %d :Train----------------------\r\n \t\t   Accuracy \t F_1 Score \r\n Against \t %.4f  \t%.4f  \r\n Favor   \t %.4f  \t%.4f   \r\n None   \t %.4f\r\n" % (epoch, acc_A_Train, F_1_against_Train, acc_F_Train, F_1_favor_Train, acc_N_Train)
            str3="Epoch %d:Test----------------------\r\n \t\t   Accuracy \t F_1 Score \r\n Against \t %.4f  \t%.4f  \r\n Favor   \t %.4f  \t%.4f   \r\n None   \t %.4f\r\n" % (epoch, acc_A_Test, F_1_against_Test, acc_F_Test, F_1_favor_Test, acc_N_Test)
            strs = np.array([str1, str2, str3])
            for string in strs:
                print(string)
                logging.info(string)

            if (F_1_favor_Test + F_1_against_Test) * 0.5 > best_F1_avg:
                best_acc, best_loss, best_F1_avg, best_F1_against, best_F1_favor, best_epoch = Test_Acc, Test_Loss, (F_1_against_Test + F_1_favor_Test) * 0.5, F_1_against_Test, F_1_favor_Test, epoch
                if args.out > 0:
                    model.save(model_out_file)
            if Test_Acc>best_accuracy:
                best_accuracy=Test_Acc
                best_epoch_accuracy=epoch

            #For Drawing
            train_accuracy_display[(0, epoch+1)] = Train_Acc
            train_accuracy_display[(1, epoch+1)] = acc_N_Train
            train_accuracy_display[(2, epoch+1)] = acc_A_Train
            train_accuracy_display[(3, epoch+1)] = acc_F_Train
            train_F1[(0, epoch+1)] = F_1_favor_Train
            train_F1[(1, epoch+1)] = F_1_against_Train
            train_F1[(2, epoch+1)] = (F_1_favor_Train + F_1_against_Train) * 0.5

            test_accuracy_display[(0, epoch+1)] = Test_Acc
            test_accuracy_display[(1, epoch+1)] = acc_N_Test
            test_accuracy_display[(2, epoch+1)] = acc_A_Test
            test_accuracy_display[(3, epoch+1)] = acc_F_Test
            test_F1[(0, epoch+1)] = F_1_favor_Test
            test_F1[(1, epoch+1)] = F_1_against_Test
            test_F1[(2, epoch+1)] = (F_1_favor_Test + F_1_against_Test) * 0.5

    str0="End. Best Epoch %d:  Test_Acc= %.4f, Test_Loss = %s, Test_F1_avg= %.4f, Test_F1_against =%.4f, Test_F1_favor = %.4f" % (best_epoch, best_acc, best_loss, best_F1_avg, best_F1_against, best_F1_favor)
    print(str0)
    logging.info(str0)
    if args.out > 0:
        print("The best NeuMF model is saved to %s" % (model_out_file))
    print("End. Best Accuracy Epoch %d:  Test_Acc= %.4f" % (best_epoch_accuracy, best_accuracy))
    logging.info("End. Best Accuracy Epoch %d:  Test_Acc= %.4f" % (best_epoch_accuracy, best_accuracy))

# Figures.
plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.5)



p2 = plt.subplot(2, 2, 1)
p2.plot(train_accuracy_display[0])
p2.plot(test_accuracy_display[0])
p2.set_xlabel("Epoch")
p2.set_ylabel("Accuracy")
p2.set_title("Accuracy vs Epoch")
p2.legend(labels=['train', 'test'], frameon=False, loc='best')

p3 = plt.subplot(2, 2, 2)
p3.plot(train_accuracy_display[1])
p3.plot(test_accuracy_display[1])
p3.set_xlabel("Epoch")
p3.set_ylabel("None_Accuracy")
p3.set_title("None_Accuracy vs Epoch")
p3.legend(labels=['train', 'test'], frameon=False, loc='best')

p4 = plt.subplot(2, 2, 3)
p4.plot(train_accuracy_display[2])
p4.plot(test_accuracy_display[2])
p4.set_xlabel("Epoch")
p4.set_ylabel("Against_Accuracy")
p4.set_title("Against_Accuracy vs Epoch")
p4.legend(labels=['train', 'test'], frameon=False, loc='best')

p5 = plt.subplot(2, 2, 4)
p5.plot(train_accuracy_display[3])
p5.plot(test_accuracy_display[3])
p5.set_xlabel("Epoch")
p5.set_ylabel("Favor_Accuracy")
p5.set_title("Favor_Accuracy vs Epoch")
p5.legend(labels=['train', 'test'], frameon=False, loc='best')

fig_path='FIGS/%s/'% args.dataset
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
plt.savefig(fig_path+'%.1f_NeuMF1_%s.png' % (time(), log_name))

plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.5)



p6 = plt.subplot(2, 2, 1)
p6.plot(train_F1[0])
p6.plot(test_F1[0])
p6.set_xlabel("Epoch")
p6.set_ylabel("F1_FAVOR")
p6.set_title("F1_FAVOR vs Epoch")
p6.legend(labels=['train', 'test'], frameon=False, loc='best')

p7 = plt.subplot(2, 2, 2)
p7.plot(train_F1[1])
p7.plot(test_F1[1])
p7.set_xlabel("Epoch")
p7.set_ylabel("F1_AGAINST")
p7.set_title("F1_AGAINST vs Epoch")
p7.legend(labels=['train', 'test'], frameon=False, loc='best')

p8 = plt.subplot(2, 2, 3)
p8.plot(train_F1[2])
p8.plot(test_F1[2])
p8.set_xlabel("Epoch")
p8.set_ylabel("F1")
p8.set_title("F1 vs Epoch")
p8.legend(labels=['train', 'test'], frameon=False, loc='best')

# plt.show()
plt.savefig(fig_path+'%.1f_NeuMF2_%s.png' % (time(), log_name))
