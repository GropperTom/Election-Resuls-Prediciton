import numpy as np
import os
from datetime import datetime

##################################################
# Input elections_data
##################################################
INPUT_FNAME = os.getcwd() + "\\..\\elections_data\\elections_2009_fixed.csv"
INPUT_FNAME_BY_DATES = os.getcwd() + "\\..\\elections_data\\2006_fixed.csv"
TENSORBOARD_DIR = os.getcwd() + "tensorboard"

##################################################
#  Accuracy calculation
##################################################
def tss(a1):
    return np.sum(np.square(np.array(a1)-np.mean(np.array(a1))))

def rss(a1, a2):
    return np.sum(np.square(np.array(a1) - np.array(a2)))

def accuracy_percent_arr(y, y_):
    """
    :param y:   numpy vector with values expected
    :param y_:  numpy vector with values predicted
    :return:    float accuracy percent between two vectors, based on TSS (Total Sum of Squares)
                and RSS (Residual Sum of Squares) calculations. Accuracy formula:
                TSS - RSS        RSS
                --------- = 1 -  ---
                   TSS           TSS
    """
    return (1 - (rss(y, y_) / float(tss(y)))) * 100

def accuracy_percent_single_val(y, y_):
    """
    :param y:   float value expected
    :param y_:  float_value predicted
    :return:    float accuracy percent between two values
    """
    err =  abs(y-y_) / float(y)
    return err * 100

##################################################
#  Log function
##################################################
LINEAR_REGRESSION_LOG_FNAME = "linreg_log.txt"
PERCEPTRON_LOG_FNAME = "perceptron_log.txt"
RNN_LOG_FNAME = "rnn_log.txt"

def log(line, fname = None):
    if fname is not None:
        full_path = os.getcwd() + "\\..\\elections_data\\" + fname
        with open(full_path, "a+") as f:
            f.write("%s %s\n" % (datetime.now().strftime('%Y-%d-%m %H:%M:%S'), line))
    else:
        print(line)

# ############################################################
# # Plot - 2009
# ############################################################
def draw_plot(data, prediction, title):
    df_2009 = data.file_df
    import matplotlib.pyplot as plt
    plot_y_data = []
    cols = df_2009.columns[4:]
    for i, elections_data in df_2009[cols].iterrows():
        plot_y_data.append(elections_data)
    plot_y_data = np.array(plot_y_data)

    xmax = ymax = 0
    lines_number = min(12, len(df_2009[df_2009.columns[1:-1]]))
    my_color = lambda i: plt.cm.get_cmap("hsv", lines_number*2)(i*2)
    prediction_delta_x = 4
    result_delta_x = 1
    for vote_index in range(lines_number):
        # elections_data
        ydata = df_2009[df_2009.columns[1:-1]].iloc[[vote_index]].values[0]
        xdata = range(len(ydata))
        xmax = max(xdata) if max(xdata) > xmax else xmax
        ymax = max(ydata) if max(ydata) > ymax else ymax
        plt.plot(xdata, ydata, color=my_color(vote_index), label=df_2009[['miflaga']].iloc[[vote_index]].values[0][0])
        # our prediction:
        plot_y_prediction = prediction[vote_index]
        plt.plot([max(xdata)+prediction_delta_x], plot_y_prediction[0], marker='o', color=my_color(vote_index))
        # votes result:
        plot_y_vote_result = df_2009[['result']].iloc[vote_index][0]
        plt.plot([max(xdata)+result_delta_x], [plot_y_vote_result], marker='o', color=my_color(vote_index))


    plt.axis([0, xmax + max(prediction_delta_x, result_delta_x) * 2, 0, ymax + 100])
    plt.xticks([max(xdata)+result_delta_x, max(xdata)+prediction_delta_x], ("result", "prediction"))

    plt.xlabel('week')
    plt.ylabel('votes')
    plt.title(title)
    plt.legend()
    plt.show()
    print("Done!")