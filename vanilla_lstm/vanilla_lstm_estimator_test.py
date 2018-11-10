import tensorflow as tf

from vanilla_lstm import vanilla_lstm_estimator


class VanillaLstmEstimatorTest(tf.test.TestCase):
    def test_estimator(self):
        self.assertTrue(True)
        output_dir = tf.test.get_temp_dir() + "/output"
        vanilla_lstm_estimator.main([None, "--output_dir=" + output_dir])


if __name__ == "__main__":
    tf.test.main()
