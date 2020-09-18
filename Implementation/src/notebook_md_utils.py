import numpy as np


def get_md_data_distribution_string(y_g1, y_g2):
    """Returns Markdown string to show table of data distribution after split in informative and non-informative
    """
    header = "|   | informativ | nicht-informativ | gesamt    |\n|:--|:--------- :|:----------------:|:---------:|\n"
    row1 = "|G1 " + _get_row_data_distribution(y_g1, y_g2)
    row2 = "|G2 " + _get_row_data_distribution(y_g2, y_g1)

    return header + row1 + row2


def _get_row_data_distribution(y_g1, y_g2):
    """Helper method for get_md_data_distribution_string
    """
    g1_distribution = np.bincount(y_g1)

    column1 = "| %i (%i%s)  " % (g1_distribution[1], round(100 / len(y_g1) * g1_distribution[1]), '%')
    column2 = "| %i (%i%s)" % (g1_distribution[0], round(100 / len(y_g1) * g1_distribution[0]), '%')
    column3 = "| %i (%i%s)|\n" % (len(y_g1), round(100 / (len(y_g2) + len(y_g1)) * len(y_g1)), '%')

    return column1 + column2 + column3


def get_md_mean_accuracy_grid(scores: dict):
    """Returns Markdown string to show table of mean accuracy
    """
    header = "| | "
    g1 = " | G1 | "
    g2 = " | G2 | "
    for key, value in scores.items():
        header += key + "| "
        acc1 = value['mean_score_g1'] * 100
        acc2 = value['mean_score_g2'] * 100
        g1 += " %.2f  | " % acc1
        g2 += " %.2f  | " % acc2
    header += "\n|:--|:--:|:--:|:--:|:--:|:--:|\n"
    return header + g1 + "\n" + g2


def get_md_test_accuracy_grid(scores: dict):
    """Returns Markdown string to show table of accuracy of test
    """
    header = "| | "
    g1 = " | Exp1 | "
    g2 = " | Exp2 | "
    avg = " | Mean | "
    for key, value in scores.items():
        header += key + "| "
        acc1 = value['accuracy_g1'] * 100
        acc2 = value['accuracy_g2'] * 100
        curr_avg = (acc1 + acc2)/2
        g1 += " %.2f  | " % acc1
        g2 += " %.2f  | " % acc2
        avg += " %.2f | " % curr_avg
    header += "\n|:--|:--:|:--:|:--:|:--:|:--:|\n"
    return header + g1 + "\n" + g2 + "\n" + avg


def get_md_confusion_matrix_grid(confusion_matrix):
    """Returns Markdown string to show confusion matrix
    """
    header = "||| Actual ||\n|:--|:--:|:--:|:--:|\n||| informativ | nicht-informativ |\n|||||\n"
    # careful, at own data, non-informative is first entry
    tn, fp, fn, tp = confusion_matrix.ravel()
    informative_predicted = "|**Predicted**| informativ | %i | %i |" % (tp, fp)
    non_informative_predicted = "|| nicht-informativ | %i | %i |" % (fn, tn)
    return header + informative_predicted + "\n" + non_informative_predicted
