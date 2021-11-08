import tensorflow as tf

import os
import sys
# ys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__),  '../src/') ))
v = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/'))
print("!!", v, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, v)

import zero_out_op

tf.compat.v1.disable_eager_execution()


class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        # zero_out_module = tf.load_op_library('./zero_out.so')
        with self.test_session():
            result = zero_out_op.zero_out([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])


if __name__ == "__main__":
    tf.test.main()
