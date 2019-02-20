
import tensorflow as tf
from elections_data_dates import PollsDataDates
from common import *
from datetime import timedelta
from time import time

CURR_LOG_NAME = RNN_LOG_FNAME

LAYERS_TYPES = ["RNN"]
LOSS_TYPES = ["MSE"] #, "CROSS_ENTROPY"]
ITERATIONS_OPTIONS = [10000, 30000, 80000]
LEARNING_RATE_OPTIONS = [0.0001, 0.001]
TRAIN_PERCENT_OPTIONS = [0.8]
DROPOUT_OPTIONS = [None]
NEURONS_NUMBER_OPTIONS = [10, 64, 128, 512]

DEFAULT_PARAMS = {
    "dropout_percent": None,
    "neurons_num": 20,
    "learning_rate": 0.0001,
    "iterations_number": 30000,
    "loss_type": "MSE",
    "train_percent": 0.8
}

def get_loss(loss_type, z, y_, W):
    losses = {"MSE":  tf.reduce_mean(tf.pow(z - y_, 2), name="MSE_loss"),
              "CROSS_ENTROPY": tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(z), reduction_indices=[1]), name="xent_loss")}
    return losses[loss_type]

class ModelRNN(object):
    def __init__(self, dropout_percent, neurons_num, loss_type, learning_rate, hidden_layers, layer_sz, run_counter):
        with tf.name_scope("RNN_"+str(run_counter)):
            self.x = tf.placeholder(tf.float32, [None, hidden_layers, layer_sz], name="x_data")
            self.y_ = tf.placeholder(tf.float32, [None, layer_sz], name="y_data")

            with tf.name_scope("OUR_LSTM"):
                with tf.name_scope("LSTM_CELL"):
                    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(neurons_num)
                    if dropout_percent is None:
                        dropoutRNN = LSTM_cell
                    else:
                        dropoutRNN = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell,output_keep_prob=dropout_percent)

                self.output, _ = tf.nn.dynamic_rnn(dropoutRNN, self.x, dtype=tf.float32)
                self.output = tf.transpose(self.output, [1, 0, 2])

                self.last = self.output[-1]

            self.W = tf.Variable(tf.truncated_normal([neurons_num, layer_sz], stddev=0.1), name="weights")
            self.b = tf.Variable(tf.ones([1, layer_sz]) * 0.1, name="biases")

            tf.summary.histogram("Weights", self.W)
            tf.summary.histogram("Biases", self.b)

            self.z = tf.add(tf.matmul(self.last, self.W), self.b, name="output")

            self.loss = get_loss(loss_type, self.z, self.y_, self.W)
            tf.summary.scalar(loss_type+ "_loss", self.loss)

            with tf.name_scope("train"):
                self.update = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


def rnn_only(data, dropout_percent = DEFAULT_PARAMS["dropout_percent"], neurons_num = DEFAULT_PARAMS["neurons_num"],
             learning_rate = DEFAULT_PARAMS["learning_rate"], iterations_number=DEFAULT_PARAMS["iterations_number"],
             loss_type=DEFAULT_PARAMS["loss_type"], train_percent=DEFAULT_PARAMS["train_percent"], run_counter = "default"):
    sess = None
    tf.reset_default_graph()

    try:
        hidden_layers = data.ngram_sz
        layer_sz = data.parties_num
        parties_num = data.train.y.cols

        model = ModelRNN(dropout_percent, neurons_num, loss_type, learning_rate, hidden_layers, layer_sz, run_counter)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        start_time = time()

        writer = tf.summary.FileWriter(TENSORBOARD_DIR)
        writer.add_graph(sess.graph)

        losses = []
        tf_predictions = tf.placeholder(tf.float32, [len(data.test.y.values),1,  layer_sz], name="tf_predictions")
        predictions = []
        accuracy = 100 - tf.reduce_mean(tf.pow(data.test.y.values.reshape(len(data.test.y.values),1,layer_sz) - tf_predictions, 2))
        tf.summary.scalar("Accuracy", accuracy)
        merged_summ = tf.summary.merge_all()
        for iteration in range(iterations_number):
            _update, _output, _W, _b, _z, _loss = sess.run(
                             fetches=[model.update, model.output, model.W, model.b, model.z, model.loss],
                             feed_dict={model.x: data.train.x.values.reshape(data.train.samples_num, hidden_layers, layer_sz),
                                        model.y_: data.train.y.values.reshape(data.train.samples_num, layer_sz)})
            if iteration % 50 == 0:
                for i, test_sample in enumerate(data.test.x.values):
                    predictions.append(model.z.eval(session=sess, feed_dict={
                        model.x: test_sample.reshape(1, hidden_layers, layer_sz)}))
                summ = sess.run(merged_summ, feed_dict={model.x: data.train.x.values.reshape(data.train.samples_num, hidden_layers, layer_sz),
                                model.y_: data.train.y.values.reshape(data.train.samples_num, layer_sz),tf_predictions: predictions})
                writer.add_summary(summ, iteration)
            losses.append(_loss)
            predictions = []

        for i, test_sample in enumerate(data.test.x.values):
            predictions.append(model.z.eval(session=sess, feed_dict={model.x: test_sample.reshape(1, hidden_layers, layer_sz)}))
            # print("Data sample: %s" % test_sample)
            # print("Expected: %s" % data.test.y.get_record(i))
            # print("Predictions: %s" % predictions)

        end_time = time()
        # Summary of predictions.
        actual_prediction = model.z.eval(session=sess, feed_dict={model.x: data.last_ngram.reshape(1, hidden_layers, layer_sz)})[0]
        #plot_polls_days = data.dates
        plot_polls_data = {data.get_party_name(pid): {"polls_data": data.get_polls_data(pid),
                                                      "actual_result": data.get_actual_result(pid),
                                                      "prediction": actual_prediction[pid],
                                                      "accuracy_mse": 100 - np.mean((data.test.y.values - predictions)**2),
                                                      # "accuracy_rss_tss": accuracy_percent_arr(data.actual_results, predictions),
                                                      }
                           for pid in range(parties_num)}
        data.accuracy_mse = 100 - np.mean((data.test.y.values - predictions)**2)
        run_data = {
                    "Acurracy_MSE": 100 - np.mean((data.test.y.values - predictions)**2),
                    "Iterations": iterations_number,
                    "Learning_Rate": learning_rate,
                    "Train_Percent": train_percent,
                    "Loss_Type": loss_type,
                    "Hidden_Layers": hidden_layers,
                    "Layer_Size": layer_sz,
                    "Neurons_Number": neurons_num,
                    "Use_Dropout": dropout_percent,
                    "Average_Accuracy": np.array([v['accuracy_mse'] for k,v in plot_polls_data.items()]).mean(),
                    "Prediction": actual_prediction,
                    "Time_elapsed": str(timedelta(seconds=(end_time -start_time))),
                    "Last_Loss": _loss,
                    "Losses_Progress": losses
                    }
        log("\n**************************************************************", CURR_LOG_NAME)
        log("\n Plot data: %s" % "\n".join(["\t%s: %s" % (k,v) for k,v in plot_polls_data.items()]), CURR_LOG_NAME)
        log("\n Run data: %s" % "\n".join(["\t%s: %s" % (k,v) for k,v in run_data.items()]), CURR_LOG_NAME)
        log("\n**************************************************************", CURR_LOG_NAME)

        sess.close()
        return plot_polls_data, run_data

    except Exception as e:
        log("\n**************************************************************", CURR_LOG_NAME)
        log("\nException-number-%s" % run_counter, CURR_LOG_NAME)
        log("\nRun number %s, do: %s, nn: %s, lr: %s, itn: %s, lt %s, tp: %s. " %
            (run_counter, dropout_percent, neurons_num, learning_rate, iterations_number, loss_type, train_percent), CURR_LOG_NAME)
        log("\nException message: \n", CURR_LOG_NAME)
        log(e, CURR_LOG_NAME)
        log("\n**************************************************************", CURR_LOG_NAME)
        if sess is not None:
            sess.close()
        raise e

##########################################################################
#                     M A I N
##########################################################################

fdata = PollsDataDates(INPUT_FNAME_BY_DATES, train_percent=DEFAULT_PARAMS['train_percent'])

############# Single run ################
# plot_polls_data, run_data = rnn_only(fdata)
# print("Time elapsed: %s, loss: %s, accuracy_mse: %s" % (run_data["Time_elapsed"], run_data["Last_Loss"], fdata.accuracy_mse))

########################### Run all options
common_counter = len(DROPOUT_OPTIONS) * len(NEURONS_NUMBER_OPTIONS) * len(LEARNING_RATE_OPTIONS) * len(ITERATIONS_OPTIONS) * len(LOSS_TYPES) * len(TRAIN_PERCENT_OPTIONS)
print("Start: summary %d runs" % common_counter)
cnt = 0
for do in DROPOUT_OPTIONS:
    for nn in NEURONS_NUMBER_OPTIONS:
        for lr in LEARNING_RATE_OPTIONS:
            for itn in ITERATIONS_OPTIONS:
                for lt in LOSS_TYPES:
                    for tp in TRAIN_PERCENT_OPTIONS:
                        cnt += 1
                        print("Start run number %s (from %s), do: %s, nn: %s, lr: %s, itn: %s, lt %s, tp: %s. " %
                              (cnt, common_counter, do, nn, lr, itn, lt, tp))
                        try:
                            plot_polls_data, run_data = rnn_only(data=fdata, dropout_percent=do, neurons_num=nn, learning_rate=lr, iterations_number=itn,loss_type=lt, train_percent=tp, run_counter=cnt)
                            print("Time elapsed: %s, loss: %s, accuracy: %s" % (run_data["Time_elapsed"], run_data["Last_Loss"],fdata.accuracy_mse))
                        except BaseException as e:
                            print("Exception see the log for the string 'Exception-number-%s'" % cnt)
