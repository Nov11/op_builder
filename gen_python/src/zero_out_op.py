import tensorflow as tf

zero_out_module = tf.load_op_library('../../out/library/libcustom_op.so')

zero_out = zero_out_module.zero_out

if __name__ == '__main__':
    print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
# array([[1, 0], [0, 0]], dtype=int32)
