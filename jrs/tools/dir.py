import shutil
import os
import time
def mkdir_if_not_exist(out_put_dir):
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

def clear(rootdir):
    filelist = os.listdir(rootdir)  # 列出该目录下的所有文件名
    for f in filelist:
        filepath = os.path.join(rootdir, f)  # 将文件名映射成绝对路劲
        if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
            os.remove(filepath)  # 若为文件，则直接删除
            print(str(filepath) + " removed!")
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)  # 若为文件夹，则删除该文件夹及文件夹内所有文件
            print("dir " + str(filepath) + " removed!")

    # os.rmdir(rootdir)

def glob_cross_validation_files(dir,nb_fold=5,ind_fold=1):
    '''
    :param dir:
    :param nb_fold: 一个fold是样本数量
    :param ind_fold:
    :return:
    '''
    train_files=sort_glob(dir)
    del train_files[(ind_fold - 1) * nb_fold:(ind_fold - 1) * nb_fold + nb_fold]
    test_files = sort_glob(dir)
    test_fiels=[test_files[i] for i in range((ind_fold - 1) *nb_fold, (ind_fold - 1) * nb_fold + nb_fold)]
    return train_files,test_fiels

def mk_or_cleardir(out_put_dir):
    """

    :rtype: object
    """
    print(out_put_dir)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    else:
        clear(out_put_dir)
def mkcleardir(out_put_dir):
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    else:
        clear(out_put_dir)
        # os.makedirs(out_put_dir)
        # shutil.rmtree(out_put_dir)
        shutil.rmtree(out_put_dir, True)  # 最后删除img总文件夹
        time.sleep(3)
        os.makedirs(out_put_dir)

def mkoutputname(dir,modelId,atlasId,targetId,type):
    return os.path.join(dir,"%s_atlas_%s_target_%s_%s"%(modelId,atlasId,targetId,type))


def filename(path,extension=False):

    tmp=os.path.basename(path)
    terms=tmp.split('.')
    return terms[0]
def listdir(dir):
    tmp=os.listdir(dir)
    tmp.sort()
    # tmp=tmp.sort()
    target_files = [os.path.join(dir, i) for i in tmp]
    return  target_files

def writeListToFile(file_list,out_put_file):
    if os.path.exists(out_put_file):
        os.remove(out_put_file)
    f=open(out_put_file,'w')

    f.writelines([line+'\n' for line in file_list])
    f.close()

import glob
from natsort import natsorted
def sort_glob(dir):
    files=glob.glob(dir)
    files.sort()
    return files

def invert_sort_glob(dir):
    files=glob.glob(dir)
    files.sort()
    return files[::-1]

def natsort_glob(dir):
    files=glob.glob(dir)

    return  natsorted(files)

import re
def sortKeyFunc(s):
    return int(os.path.basename(s)[:-7])

from pathlib import Path

def sort_globV2(dir):
    files=glob.glob(dir)
    files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(x))])
    return files
def sort_time_glob(dir):
    files=glob.glob(dir)
    files.sort(key=os.path.getmtime)
    return files


def get_name_wo_suffix(path):
    return os.path.basename(path).split('.')[0]

def copy_dir(src,dst):
    print(f"copy {src}===>{dst}")
    mk_or_cleardir(dst)
    for root, dirs, files in os.walk(src):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, dst)
            # print(src_file)


def parent_dir(path):
    if path.endswith('/'):
        path=path[:-2]
    return os.path.dirname(path)