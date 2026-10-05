"""Actual TensorFlow checks, skipped only when the training extra is absent."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

HAS_TF = importlib.util.find_spec('tensorflow') is not None


@unittest.skipUnless(HAS_TF, 'Install the train extra for model tests')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf
        tf.keras.utils.set_random_seed(42)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    def test_architecture_variants_and_gradients(self):
        from busi_segmentation.model import unet_model, dice_loss, iou_loss
        tf = self.tf
        x = tf.random.uniform((2, 32, 32, 1))
        y = tf.cast(x > .5, tf.float32)
        for options in ({}, {'residual': True}, {'attention': True},
                        {'activation': 'swish'}, {'residual': True, 'attention': True, 'activation': 'swish'}):
            with self.subTest(options=options):
                tf.keras.backend.clear_session()
                model = unet_model(input_shape=(32, 32, 1), base_filters=4, **options)
                with tf.GradientTape() as tape:
                    output = model(x, training=True)
                    loss = dice_loss(y, output) + iou_loss(y, output)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertEqual(output.shape, y.shape)
                self.assertTrue(np.isfinite(output.numpy()).all())
                self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
                self.assertTrue(np.any(output.numpy() > 0) and np.all(output.numpy() < 1))

    def test_metric_is_image_weighted(self):
        from busi_segmentation.model import hard_dice_per_image
        metric = self.tf.keras.metrics.MeanMetricWrapper(hard_dice_per_image)
        truth = self.tf.ones((5, 2, 2, 1))
        metric.update_state(truth, self.tf.zeros_like(truth))
        metric.update_state(self.tf.zeros((1, 2, 2, 1)), self.tf.zeros((1, 2, 2, 1)))
        self.assertAlmostEqual(float(metric.result()), 1/6, places=6)

    def test_checkpoint_round_trip(self):
        from busi_segmentation.model import unet_model
        tf = self.tf
        tf.keras.backend.clear_session()
        model = unet_model(input_shape=(32, 32, 1), base_filters=4)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        x = tf.random.uniform((2, 32, 32, 1))
        model.train_on_batch(x, tf.cast(x > .5, tf.float32))
        expected = model(x, training=False).numpy()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.weights.h5'
            model.save_weights(checkpoint)
            restored = unet_model(input_shape=(32, 32, 1), base_filters=4)
            restored.load_weights(checkpoint)
            np.testing.assert_allclose(restored(x, training=False).numpy(), expected, atol=1e-6)

    def test_original_width_forward_pass(self):
        from busi_segmentation.model import unet_model
        self.tf.keras.backend.clear_session()
        model = unet_model(base_filters=64)
        output = model(self.tf.zeros((1, 128, 128, 1)), training=False)
        self.assertEqual(output.shape, (1, 128, 128, 1))
        self.assertTrue(np.isfinite(output.numpy()).all())

    def test_invalid_shapes_rejected(self):
        from busi_segmentation.model import unet_model
        with self.assertRaises(ValueError):
            unet_model(input_shape=(127, 128, 1))
        with self.assertRaises(ValueError):
            unet_model(num_classes=2)


if __name__ == '__main__':
    unittest.main()
