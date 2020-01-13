"""
    File 'plotting.py' has functions for plotting different data.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

# Matplotlib color palette with 500 colors
color_palette = plt.cm.get_cmap('jet', 500)


def hist_plotting(data, labels, title,
                  width=1920, height=1080, dpi=96, font_size=21):
    """
        Method to plot histogram.
        param:
            1. data - numpy array of data that should be plotted
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. width - int value of plot width in pixels (1920 as default)
            5. height - int value of plot height in pixels (1080 as default)
            6. dpi - int value of plot dpi (96 as default)
            7. font_size - int value of text size on plot (21 as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.hist(data, bins=round(math.sqrt(len(data))))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plots/data distributions/" + title + ".png", dpi=dpi)
    plt.close()

    return


def bar_plotting(data, labels, title, folder,
                 width=1920, height=1080, dpi=96, font_size=16, color='b'):
    """
        Method to plot barchart.
        param:
            1. data - vector (2) of data that should be plotted
                and ticks for x axis
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. folder - string path to save plot
            5. width - int value of plot width in pixels (1920 as default)
            6. height - int value of plot height in pixels (1080 as default)
            7. dpi - int value of plot dpi (96 as default)
            8. font_size - int value of text size on plot (16 as default)
            9. color - string value of color name for plot ('b' as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.bar((np.arange(len(data[1])))[::-1], data[0], align='center',
            color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(np.arange(len(data[1]))[::-1], data[1], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plots/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def data_plotting(data, title, folder,
                  width=1920, height=1080, dpi=96, font_size=21, color='k'):
    """
        Method to plot 3D plots of dataset values.
        param:
            1. data - pandas DataFrame that will be plotted
                by first 3 features
            2. title - string name of plot
            3. folder - string path to save plot
            4. width - int value of plot width in pixels (1920 as default)
            5. height - int value of plot height in pixels (1080 as default)
            6. dpi - int value of plot dpi (96 as default)
            7. font_size - int value of text size on plot (21 as default)
            8. color - string value of color name for plot ('k' as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    ax = plt.axes(projection='3d')
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],
               cmap=color_palette, c=color)
    ax.set_title(title)
    plt.savefig("plots/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def line_plotting(data, labels, title, folder,
                  width=1920, height=1080, dpi=96, font_size=21):
    """
        Method to Elbow (Knee), Silhouette methods for k-means and
            Elbow (Knee) method for DMDBSCAN distance choosing algorithm.
        param:
            1. data - vector (3) of  data that should be plotted, data for
                x axis and integer value for optimal parameter
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. folder - string path to save plot
            5. width - int value of plot width in pixels (1920 as default)
            6. height - int value of plot height in pixels (1080 as default)
            7. dpi - int value of plot dpi (96 as default)
            8. font_size - int value of text size on plot (21 as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.plot(data[1], data[0], 'bx-')
    plt.vlines(data[2], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plots/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return
