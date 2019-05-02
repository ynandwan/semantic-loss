"""Cifar-10 Train/Eval module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
from model import Net, HParams
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("mode", "train", "train or eval.")
tf.app.flags.DEFINE_string("train_data_path", "", "Filepattern for training data.")
tf.app.flags.DEFINE_string("eval_data_path", "", "Filepattern for eval data")
tf.app.flags.DEFINE_integer("image_size", 32, "Image side length.")
tf.app.flags.DEFINE_string("train_dir", "", "Directory to keep training outputs.")
tf.app.flags.DEFINE_string("eval_dir", "", "Directory to keep eval outputs.")
tf.app.flags.DEFINE_integer("batch_size", 256, "The batch size for training.")
tf.app.flags.DEFINE_integer("eval_batch_count", 100, "Number of batches to eval.")
tf.app.flags.DEFINE_bool("eval_once", False, "Whether evaluate the model only once.")
tf.app.flags.DEFINE_string(
    "log_root", "", "Directory to keep the checkpoints. Should be a " "parent directory of FLAGS.train_dir/eval_dir."
)
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus used for training. (0 or 1 or 2)")


def train(hps):
    """Training loop."""
    images, labels = cifar_input.build_input(FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
    model = Net(hps, images, labels, FLAGS.mode)
    model.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    )
    sys.stdout.write("total_params: %d\n" % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS
    )

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    labeled_examples = tf.greater(tf.reduce_max(model.labels, axis=1), tf.zeros([hps.batch_size, 1]))
    labeled_examples = tf.cast(labeled_examples, tf.float32)
    correct_predictions = tf.cast(tf.equal(predictions, truth), tf.float32)
    correct_predictions = tf.multiply(correct_predictions, labeled_examples)
    precision = tf.reduce_sum(correct_predictions) / tf.reduce_sum(labeled_examples)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge([model.summaries, tf.summary.scalar("Precision", precision)]),
    )

    logging_hook = tf.train.LoggingTensorHook(
        tensors={
            "step": model.global_step,
            "loss": model.cost,
            "wmc": model.wmc,
            "cross_entropy": model.cross_entropy,
            "precision": precision,
        },
        every_n_iter=100,
    )

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.1

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(model.global_step, feed_dict={model.lrn_rate: self._lrn_rate})

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 10000:
                self._lrn_rate = 0.1
            elif train_step < 20000:
                self._lrn_rate = 0.05
            elif train_step < 35000:
                self._lrn_rate = 0.01
            else:
                self._lrn_rate = 0.001

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.log_root,
        hooks=[logging_hook, _LearningRateSetterHook()],
        chief_only_hooks=[summary_hook],
        save_summaries_steps=0,
        config=tf.ConfigProto(allow_soft_placement=True),
    ) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)


def evaluate(hps):
    """Eval loop."""
    images, labels = cifar_input.build_input(FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
    model = Net(hps, images, labels, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error("Cannot restore checkpoint: %s", e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info("No model to eval yet at %s", FLAGS.log_root)
            continue
        tf.logging.info("Loading checkpoint %s", ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step, wmc) = sess.run(
                [model.summaries, model.cost, model.predictions, model.labels, model.global_step, model.wmc]
            )

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(tag="Precision", simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(tag="Best Precision", simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info(
            "loss: %.5f, precision: %.5f, wmc: %.5f, best precision: %.5f" % (loss, precision, wmc, best_precision)
        )
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(60)


def main(_):
    if FLAGS.num_gpus >= 1 and FLAGS.mode == "train":
        dev = "/gpu:0"
    elif FLAGS.num_gpus >= 2 and FLAGS.mode == "eval":
        dev = "/gpu:1"
    else:
        dev = "/cpu:0"

    if FLAGS.mode == "train":
        batch_size = FLAGS.batch_size
    elif FLAGS.mode == "eval":
        batch_size = 100

    num_classes = 10

    hps = HParams(
        batch_size=batch_size,
        num_classes=num_classes,
        min_lrn_rate=0.0001,
        lrn_rate=0.1,
        num_units=5,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer="mom",
    )

    with tf.device(dev):
        if FLAGS.mode == "train":
            train(hps)
        elif FLAGS.mode == "eval":
            evaluate(hps)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
