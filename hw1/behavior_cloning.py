import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

ENV_NAME = "Humanoid-v2"
with open(f"bc_data/bc_data_{ENV_NAME}", "rb") as data_file:
    raw_data = pickle.load(data_file)

x_train = raw_data[0]
y_train_raw = raw_data[1]
y_train_raw = y_train_raw.squeeze()

state_space = 376
action_space = 17

def create_model():
    tf.reset_default_graph()
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_space])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_space])

    w0 = tf.get_variable(name='W0', shape=[state_space, 32],  initializer=tf.contrib.layers.xavier_initializer())
    w1 = tf.get_variable(name='W1', shape=[32, 48],           initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable(name='W2', shape=[48, action_space], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name="b0", shape=[32], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name="b1", shape=[48], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name="b2", shape=[action_space], initializer=tf.constant_initializer(0.))

    weights = [w0, w1, w2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W)
        layer = layer + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer
    return input_ph, output_ph, output_pred

input_ph, output_ph, output_pred = create_model()
with tf.Session() as sess:
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()

    batch_size = 32
    for training_step in range(10001):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(x_train), size=batch_size)
        input_batch = x_train[indices]
        output_batch = y_train_raw[indices]

        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})
        # print the mse every so often
        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
            saver.save(sess, '/tmp/model.ckpt')


input_ph, output_ph, output_pred = create_model()
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/model.ckpt")

    env = gym.make(ENV_NAME)
    max_steps = 10000

    for i in range(200):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            actions = output_pred_run = sess.run(output_pred, feed_dict={input_ph: (obs,)})
            action = np.argmax(actions)
            print(action)
            obs, r, done, _ = env.step(action)
            env.render()
            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
