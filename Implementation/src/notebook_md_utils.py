import numpy as np


def get_md_data_distribution_string(y_g1, y_g2):
    """Returns Markdown string to show data distribution after split in informative and non-informative
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
