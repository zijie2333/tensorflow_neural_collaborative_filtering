'''
Created on Aug 8, 2016
Processing datasets.
@author: Zijie Huang
'''
import scipy.sparse as sp
import numpy as np


####For grouped friendship....friends are index.
class Dataset_New(object):

    def __init__(self, args):
        '''
        Constructor
        '''
        self.num_items = args.num_items
        self.num_classes = args.num_classes
        self.num_users = args.num_users

        Train_data = args.path + args.dataset + "/Train_Athesim_5_Renumbered_000_own.txt"
        Test_data =  args.path +  args.dataset+ "/Test_Athesim_5_Renumbered_000_own.txt"

        self.largest_friends_number = self.Find_largest_friends_number(Train_data,Test_data)


        self.train_users, self.train_items, self.train_labels, self.train_isintargets,self.train_weights = self.load_train_file_as_numpy(
           Train_data,self.largest_friends_number)
        self.test_users, self.test_labels, self.test_isintargets,self.test_weights= self.load_test_file_as_numpy(
            Test_data,self.largest_friends_number)
        self.t_train_users, self.t_train_labels, self.t_train_isintargets,self.t_train_weights = self.load_test_file_as_numpy(
            Train_data,self.largest_friends_number)

    def Find_largest_friends_number(self,Train_data,Test_data):
        largest_friends_number=0

        for filename in [Train_data,Test_data]:
            with open(filename,'r') as f:
                for line in f:
                    if len(line)<3:
                        continue
                    line = line.rstrip("\r\n")
                    arr = line.split("\t")
                    friends=arr[3].split(":")
                    if len(friends[2:])==0:
                        continue ##no friends
                    largest_friends_number=max(largest_friends_number,len(friends[2:]))

        print("Largest friends number is %d"%largest_friends_number)
        return largest_friends_number



    def convert_stances(self, stance):
        #convert stance into np.array of shape [1]
        stance_ID=np.zeros(1,dtype="int32")

        if stance == 'NONE':
            stance_ID[0] = 0
        elif stance == 'AGAINST':
            stance_ID[0]= 1
        elif stance == 'FAVOR':
            stance_ID[0] = 2
        return stance_ID


    def load_train_file_as_numpy(self, filename, largest_friends_number):   #original_num*3
        labels = []
        users = []
        items = []
        user_weights=[]
        isintargets = []
        total_users=set()
        total_tweet=list()
        favor,against,none=0,0,0
        with open(filename, "r") as f:
            for line in f:
                if len(line)<3:
                    continue
                line = line.rstrip("\r\n")
                arr = line.split("\t")
                friends=arr[3].split(":")
                if len(friends[2:])==0:
                    continue ##no friends

                user_friends = []
                user = arr[2]
                if user in total_users:  #if user in total_users:  # use only the first stance of a user.
                    continue
                total_users.add(user)
                total_tweet.append(user)

                ##padding friendship.
                for friend in friends[2:]:
                    friend_ID = int(friend)
                    user_friends.append(int(friend_ID-1))
                if len(user_friends) != largest_friends_number:
                    padding = np.zeros(largest_friends_number - len(user_friends), dtype='int32').tolist()
                    user_friends = np.asarray(user_friends + padding)
                else:
                    user_friends=np.asarray(user_friends)

                user_weight_0 = np.ones(len(friends[2:]))
                user_weight_1 = np.zeros(largest_friends_number-len(friends[2:]))
                user_weight = np.concatenate((user_weight_0,user_weight_1))



                ##create three stances
                isintarget=arr[5]
                stance=self.convert_stances(arr[4])
                #Calculate f,a,n portion.
                if stance[0] == 0:
                    none+=1
                elif stance[0] == 1:
                    against+=1
                elif stance[0] == 2:
                    favor+=1

                for x in xrange(0,3):
                    if x == stance[0]:
                        y=1
                    else:
                        y=0
                    item_now=np.zeros(1,dtype='int32')
                    item_now[0]=x
                    label_now=np.zeros(1,dtype='int32')
                    label_now[0]=y

                    users.append(user_friends)
                    user_weights.append(user_weight)
                    items.append(item_now)
                    labels.append(label_now)
                    isintargets.append(isintarget)


        ####reshape
        items=np.reshape(np.array(items),(-1,1))
        labels = np.reshape(np.array(labels), (-1, 1))

        total=favor+against+none
        print("NUmber of total user is %d" % len(total_users))
        print("NUmber of total tweet is %d" % len(total_tweet))
        print("favor=%f against=%f none = %f" %(favor/float(total), against/float(total),none/float(total)))
        return np.array(users), items, labels, isintargets, np.asarray(user_weights)

    def load_test_file_as_numpy(self, filename,largest_friends_number):
        labels = []
        users = []
        items = []
        user_weights= []
        isintargets = []
        total_users = set()
        total_tweet = list()
        favor, against, none = 0, 0, 0
        with open(filename, "r") as f:
            for line in f:
                if len(line)<3:
                    continue
                line = line.rstrip("\r\n")
                arr = line.split("\t")
                friends = arr[3].split(":")
                if len(friends[2:]) == 0:
                    continue  ##no friends
                user = arr[2]

                #if user in total_users: ## already have an stance
                if user in total_users:
                    continue

                total_users.add(user)
                total_tweet.append(user)
                user_friends = []
                for friend in friends[2:]:
                    friend_ID = int(friend)
                    user_friends.append(int(friend_ID - 1))
                if len(user_friends) != largest_friends_number:
                    padding = np.zeros(largest_friends_number - len(user_friends), dtype='int32').tolist()
                    user_friends = np.asarray(user_friends + padding)
                else:
                    user_friends = np.asarray(user_friends)

                user_weight_0 = np.ones(len(friends[2:]))
                user_weight_1 = np.zeros(largest_friends_number-len(friends[2:]))
                user_weight = np.concatenate((user_weight_0,user_weight_1))


                isintarget = arr[5]
                stance = self.convert_stances(arr[4])  #0,1,2
                #Calculate f,a,n portion.
                if stance[0] == 0:
                    none+=1
                elif stance[0] == 1:
                    against+=1
                elif stance[0] == 2:
                    favor+=1

                users.append(user_friends)
                user_weights.append(user_weight)
                labels.append(stance)
                isintargets.append(isintarget)
        ####reshape
        labels = np.reshape(np.array(labels), (-1, 1))
        total = favor + against + none
        print("NUmber of total user is %d" % len(total_users))
        print("NUmber of total tweet is %d" % len(total_tweet))
        print("favor=%f against=%f none = %f" % (favor / float(total), against / float(total), none / float(total)))
        return np.array(users),labels, isintargets,np.asarray(user_weights)


###n-hot vector
class Dataset(object):

    def __init__(self, args):
        '''
        Constructor
        '''
        self.num_items = args.num_items
        self.num_classes = args.num_classes
        self.num_users = args.num_users

        Train_data = args.path + args.dataset + "/Train_Athesim_5_Renumbered_000_own.txt"
        Test_data =  args.path +  args.dataset+ "/Test_Athesim_5_Renumbered_000_own.txt"



        self.train_users, self.train_items, self.train_labels, self.train_isintargets = self.load_train_file_as_numpy(
           Train_data)
        self.test_users, self.test_labels, self.test_isintargets= self.load_test_file_as_numpy(
            Test_data)
        self.t_train_users, self.t_train_labels, self.t_train_isintargets= self.load_test_file_as_numpy(
            Train_data)



    def convert_stances(self, stance):
        #convert stance into np.array of shape [1]
        stance_ID=np.zeros(1,dtype="int32")

        if stance == 'NONE':
            stance_ID[0] = 0
        elif stance == 'AGAINST':
            stance_ID[0]= 1
        elif stance == 'FAVOR':
            stance_ID[0] = 2
        return stance_ID


    def load_train_file_as_numpy(self, filename):   #original_num*3
        labels = []
        users = []
        items = []
        isintargets = []
        total_users=set()
        total_tweet=list()
        favor,against,none=0,0,0
        with open(filename, "r") as f:
            for line in f:
                if len(line)<3:
                    continue
                line = line.rstrip("\r\n")
                arr = line.split("\t")
                friends=arr[3].split(":")
                if len(friends[2:])== 0:
                    continue ##no friends

                user_friends = []
                user = arr[2]
                if user in total_users:  #if user in total_users:  # use only the first stance of a user.
                    continue
                total_users.add(user)
                total_tweet.append(user)

                ##padding friendship.
                user_friends = np.zeros(self.num_users)
                if len(friends[2:])!=0:
                    for friend in friends[2:]:
                        friend_ID = int(friend)
                        user_friends[int(friend_ID-1)]=1


                ##create three stances
                isintarget=arr[5]
                stance=self.convert_stances(arr[4])
                #Calculate f,a,n portion.
                if stance[0] == 0:
                    none+=1
                elif stance[0] == 1:
                    against+=1
                elif stance[0] == 2:
                    favor+=1

                for x in xrange(0,3):
                    if x == stance[0]:
                        y=1
                    else:
                        y=0
                    item_now=np.zeros(1,dtype='int32')
                    item_now[0]=x
                    label_now=np.zeros(1,dtype='int32')
                    label_now[0]=y

                    users.append(user_friends)
                    items.append(item_now)
                    labels.append(label_now)
                    isintargets.append(isintarget)


        ####reshape
        items=np.reshape(np.array(items),(-1,1))
        labels = np.reshape(np.array(labels), (-1, 1))

        total=favor+against+none
        print("NUmber of total user is %d" % len(total_users))
        print("NUmber of total tweet is %d" % len(total_tweet))
        print("favor=%f against=%f none = %f" %(favor/float(total), against/float(total),none/float(total)))
        return np.asarray(users), items, labels, isintargets

    def load_test_file_as_numpy(self, filename):
        labels = []
        users = []
        isintargets = []
        total_users = set()
        total_tweet = list()
        num_of_no_friend=0
        favor, against, none = 0, 0, 0
        with open(filename, "r") as f:
            for line in f:
                if len(line)<3:
                    continue
                line = line.rstrip("\r\n")
                arr = line.split("\t")
                friends = arr[3].split(":")
                if friends[1] == 0:
                    continue  ##no user

                if len(friends[2:]) == 0:
                    continue

                user = arr[2]

                #if user in total_users: ## already have an stance
                if user in total_users:
                    continue

                total_users.add(user)
                total_tweet.append(user)

                ##padding friendship.
                user_friends = np.zeros(self.num_users)
                if len(friends[2:]) != 0:
                    num_of_no_friend+=1
                    for friend in friends[2:]:
                        friend_ID = int(friend)
                        user_friends[int(friend_ID - 1)] = 1


                isintarget = arr[5]
                stance = self.convert_stances(arr[4])  #0,1,2
                #Calculate f,a,n portion.
                if stance[0] == 0:
                    none+=1
                elif stance[0] == 1:
                    against+=1
                elif stance[0] == 2:
                    favor+=1

                users.append(user_friends)
                labels.append(stance)
                isintargets.append(isintarget)
        ####reshape
        labels = np.reshape(np.array(labels), (-1, 1))
        total = favor + against + none
        print("NUmber of total user is %d" % len(total_users))
        print("NUmber of total tweet is %d" % len(total_tweet))
        print("Number of no friend is %d"%num_of_no_friend)
        print("favor=%f against=%f none = %f" % (favor / float(total), against / float(total), none / float(total)))
        return np.asarray(users),labels, isintargets


