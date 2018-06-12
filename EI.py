import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
RANDOM_SEED = 3

def get_data():
    """ Read the iris data set and split them into training and test sets """
##    plt.show()

    # split into input (X) and output (Y) variables
    df = pd.read_csv('splice.data', sep=",")
    df = df.values
##    print(df.shape)    
    target = df[:,0]
##    print(target)
    data = np.array([[0]*60]*len(df))
    for i in range(0,len(df)):
        temp = np.array([0])
        temp = [x for x in df[i,2]]
        temp = np.delete(temp, np.where(temp == ' '))
        temp = np.delete(temp, np.where(temp == ' '))
    ##    print(len(temp))
    ##    print(temp)
        numv = {'A': 0,'C': 1,'T': 2,'G': 3,'D': 4,'N': 5,'S': 6,'R': 7}
        temp = [numv[item] for item in temp]
        data[i,:] = temp
##    print(data)

    d = np.array([[0.0]*len(data[0])]*len(data))
##    print(data.shape,d.shape)
##    print(data)
    for i in range(0,(len(data[0]))):
        for j in range(0,len(data)):
            maxi = np.amax(data[:,i])
            d[j,i] = data[j,i] / maxi
            
    
    numv = {'EI': 0,'IE': 1,'N': 2}
    target = [numv[item] for item in target]
    t = np.array([[0]]*len(df))
    for i in range(0,len(df)):
        t[i,0] = target[i]
##    print(t)
    ##    print(target)
    # Prepend the column of 1s for bias
    N, M  = d.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = d

    # Convert into one-hot vectors
    num_labels = len(np.unique(t))

    all_Y = np.eye(num_labels)[target]  # One liner trick!
    
    return train_test_split(all_X, all_Y, test_size=0.3, random_state=RANDOM_SEED)

train_x,test_x,train_y,test_y = get_data()
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
n_nodes_hl1 = 100
n_nodes_hl2 = 60
n_nodes_hl3 = 30

n_classes = 3
##sizes = len(train_x)
##print(sizes)
##batch_size = 100
hm_epochs = 20000

l = tf.placeholder('float')
x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes]))}
##output_layer['weight'] *= 0.0005

# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
##    l1 = (tf.matmul(data,hidden_1_layer['weight']))    
##    l1 = tf.nn.relu(l1)
    l1 = tf.nn.sigmoid(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
##    l2 = tf.nn.relu(l2)
    l2 = tf.nn.sigmoid(l2)
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
##    l3 = tf.nn.relu(l3)
    l3 = tf.nn.sigmoid(l3)
    output = tf.add(tf.matmul(l3,output_layer['weight']),output_layer['bias'])
##    output = tf.nn.sigmoid(output)
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

##    pred = tf.nn.softmax(tf.matmul(x, hidden_1_layer['weight']) + hidden_1_layer['bias']) # Softmax
    
##    cost = tf.losses.mean_squared_error(y,prediction)
##    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
##    cost = tf.reduce_mean(tf.divide(tf.square(y-prediction),2))
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l).minimize(cost)

    errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(hm_epochs):
            epoch_loss = 0
##            i=0
##            while i < len(train_x):
##                start = i
##                end = i+batch_size
##                batch_x = np.array(train_x[start:end])
##                batch_y = np.array(train_y[start:end])
##            print(cost)

##            p2 = sess.run(prediction, feed_dict={x: train_x,y: train_y} )
##            print(p2)
            lr = 0.1
            pred = sess.run(prediction, feed_dict={x: train_x} )
            c = sess.run(cost, feed_dict={x: train_x,y: train_y} )
            sess.run(optimizer, feed_dict={x: train_x,y: train_y,l: lr} )

            errors.append(c)
            epoch_loss = c

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            train_accuracy = np.mean(np.argmax(train_y, axis=1) == np.argmax(pred, axis=1))

            test_accuracy  = np.mean(np.argmax(test_y, axis=1) == np.argmax(sess.run(prediction, feed_dict={x: test_x, y: test_y}), axis=1))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        
        p =  sess.run(prediction, feed_dict={x: train_x} )
        c = sess.run(cost, feed_dict={x: train_x,y: train_y} )
        print('Loss For training Dataset:',c)
        p2 =  sess.run(prediction, feed_dict={x: test_x} )        
        c = sess.run(cost, feed_dict={x: test_x,y: test_y} )
        print('Loss For testing Dataset:',c)

        p = np.argmax(p, axis=1)
        t = np.argmax(train_y, axis=1)

        p2 = np.argmax(p2, axis=1)
        t2 = np.argmax(test_y, axis=1)        

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == p)

        test_accuracy  = np.mean(np.argmax(test_y, axis=1) == p2)

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

##        plt.scatter(t, p)
##        plt.plot([t.min(), t.max()], [t.min(), t.max()], 'k--', lw=3)
##        plt.xlabel('Measured')
##        plt.ylabel('Predicted')
##        plt.show()
##
##        plt.scatter(t2, p2)
##        plt.plot([t2.min(), t2.max()], [t2.min(), t2.max()], 'k--', lw=3)
##        plt.xlabel('Measured')
##        plt.ylabel('Predicted')
##        plt.show()
##    
##        plt.plot(errors)
##        plt.show()

train_neural_network(x)
