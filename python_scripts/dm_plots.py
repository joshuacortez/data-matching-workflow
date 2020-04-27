import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def number_prettify(value):
    if value >= 1e9:
        if value >= 1e10:
            value = "{} B".format(int(value/1e9))
        else:
            value = "{:.1f} B".format(value/1e9)
    elif (value < 1e9) and (value >= 1e6):
        if value >= 1e7:
            value = "{} M".format(int(value/1e6))
        else:
            value = "{:.1f} M".format(value/1e6)
    elif (value < 1e6) and (value >= 1e3):
        if value >= 1e4:
            value = "{} K".format(int(value/1e3))
        else:
            value = "{:.1f} K".format(value/1e3)
    else:
        value = "{}".format(int(value))

    return value

def percent_prettify(value):
    value = value*100
    if (value < 1) and (value > -1) and (value != 0):
        value = "{:.1f}%".format(value)
    else:
        value = "{}%".format(int(value))
    return value

def plot_threshold_finder_curves(probs, threshold, recall_weight = 1, figsize = (7,5)):
    expected_matches = probs.cumsum()
    
    x = [i for i in range(len(expected_matches))]
    
    recall_weight = 1
    recall = expected_matches / expected_matches[-1]
    precision = expected_matches / np.arange(1, len(expected_matches) + 1)
    score = recall * precision / (recall + (recall_weight ** 2) * precision)
    # multiply by 2 to match f1-score
    score = score*2
    i = np.argmax(score)
    
    _, ax = plt.subplots(figsize = figsize)
    
    ax.plot(x, recall, label = "Recall (Optimal Val {})".format(percent_prettify(recall[i])))
    ax.plot(x, precision, label = "Precision (Optimal Val {})".format(percent_prettify(precision[i])))
    ax.plot(x, score, label = "F1 Score (Optimal Val {})".format(percent_prettify(score[i])))
    ax.axes.axvline(x = i, linestyle = "--", color = "grey", label = "Optimal F1 Score\n{} Matches\n{} Threshold".format(number_prettify(i),percent_prettify(probs[i])))
    ax.set_xlabel("Expected Number of Matches")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: number_prettify(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: percent_prettify(x)))
    ax.set_title("Curves for Determining Optimal Threshold")
    
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax.legend(bbox_to_anchor=(1.03,0), loc="lower left")
    plt.show()

    _, ax = plt.subplots(figsize = figsize)

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: percent_prettify(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: percent_prettify(x)))
    ax.set_title("Precision-Recall Curve")

    plt.show()

def plot_prob_histogram(probs, threshold, figsize = (7,5)):
    _, ax = plt.subplots(figsize = figsize)
    ax.hist(probs, bins = 40)
    ax.axes.axvline(x = threshold, linestyle = "--", label = "threshold = {:.1f}%".format(threshold*100))
    ax.legend()
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: percent_prettify(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: number_prettify(x)))
    ax.set_title("Histogram of Probability Scores")
    plt.show()