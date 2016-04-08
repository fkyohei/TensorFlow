import tensorflow as tf
import multiprocessing as mp

core_num = mp.cpu_count()
config = tf.ConfigProto(
    inter_op_parallelism_threads=core_num,
    intra_op_parallelism_threads=core_num )
sess = tf.Session(config=config)

hello = tf.constant('hello, tensorflow!!')
print hello
print sess.run(hello)

a = tf.constant(10)
b = tf.constant(32)
print a
print b
c = a + b   # not calculate here
print c
print sess.run(c)
