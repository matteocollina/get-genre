# -*- coding: UTF-8 -*-
import os, shutil
import ntpath
import glob
import re
import io
import json
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

__cacheFileName = "cache.txt"
__apiFileName = "api.json"
__logFileName = "log_track.txt"

def remove_audio_files_on_current_dir():
    remove_audio_files("./")
def remove_audio_files(path):
    audio_extensions = ["mp3","wav","mp4"]
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            filename, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.replace(".", "")
            if os.path.isfile(file_path) & (file_extension in audio_extensions):
                os.unlink(file_path)
        except Exception as e:
            print(e)
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_pathfile_for_name_in_current_dir(path,fileName,ext):
    '''
    :param path: search file in path
    :param fileName: name of file without extension
    :param ext: extension of the file to find
    :return: filepath
    '''
    #print("get_pathfile_for_name_in_current_dir {0}\t{1}\t{2}\t{3}".format(path,fileName,ext,glob.glob(os.path.join(path, '{0}.*'.format(fileName)))))
    for file_path in glob.glob(os.path.join(path, '{0}.*'.format(fileName))):
        filename, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.replace(".", "")
        #print("Path {0} ? {1}".format(file_path,file_extension==ext))
        if(file_extension==ext):
            return file_path

# same normalization of instantmusic (--restrict-filenames option)
def normalize(str):
    print("normalize",str)
    #str = unicodedata.normalize("NFKD", str)
    str = str.replace(" ","_").replace("&","")
    return re.sub('[^A-Za-z0-9-_]+', '', str)

def incrementVersionOnCache():
    old_v = loadVersionOnCache()
    with open(__cacheFileName, 'w') as outfile:
        new_v = old_v+1
        json_to_save = {'version':new_v}
        print("increased version in cache: {0}".format(new_v))
        json.dump(json_to_save, outfile)
def loadVersionOnCache():
    try:
        with open(__cacheFileName,'r') as f:
            data = json.load(f)
            return data['version']
    except Exception as e:
        return 0

#Get json of all api
def loadAPI():
    try:
        with open(__apiFileName,'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print("Error while reading {0}: {1}".format(__apiFileName,e))
        return 0

def resetLog():
    try:
        os.unlink(get_pathfile_for_name_in_current_dir('./', "log", 'txt'))
    except Exception:
        print("Error while deleting log file")
def log(text,filename=__logFileName):
    with open(filename, 'a') as outfile:
        outfile.write("{0}\n".format(text))

def items_in_dir(path):
    return sum([len(files) for r, d, files in os.walk(path)])

def get_confusion_matrix(data, row_labels, col_labels,title=""):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, format(data[i, j], '.2f'),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    return plt

def save_plot(plot,name):
    plot.savefig('{0}.png'.format(name))

