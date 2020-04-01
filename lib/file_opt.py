# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import zipfile
import pandas
import numpy as np


def get_files(path, file_type='', append_path=True, need_sort=False):
    """
    获取某个文件目录下全部文件或某个类别的文件名或完整的文件路径
    :param path: 文件的路径
    :param file_type: 文件的路径
    :param append_path: 用于控制是否返回完成的文件路径
    :param need_sort: 是否对文件名按字典序排序
    :return: 返回所有符合条件的文件名列表
    """
    if not isinstance(file_type, str):
        return []
    all_files = os.listdir(path)

    if not file_type:
        result = all_files
    else:
        new_file_types = [file_type.lower(), file_type.upper()]
        result = filter(lambda file_name: file_name.split(".")[-1] in new_file_types, all_files)
    if append_path:
        result = [os.path.join(path, f) for f in result]
    if need_sort:
        result.sort()
    return result


def get_multi_type_files(path, file_types=[], append_path=True, need_sort=False):
    """
    获取某个文件目录下全部文件或某个类别的文件名或完整的文件路径，无法筛选出无后缀名的文件，
    对file_types会做大小写扩展。

    Parameters:
        path - 文件的路径
        file_types - 文件类型
        append_path - 用于控制是否返回完成的文件路径
        need_sort - 是否对文件名按字典序排序

    Returns:
        file_types数组为空，返回当前目录下所有文件
        file_types参数类型不为list，返回空结果
    """
    if not isinstance(file_types, list):
        return []
    all_files = os.listdir(path)
    if len(file_types) == 0:
        result = all_files
    else:
        new_file_types = []
        for f_type in file_types:
            new_file_types.append(f_type.lower())
            new_file_types.append(f_type.upper())
        result = filter(lambda file_name: file_name.split(".")[-1] in new_file_types, all_files)
    if append_path:
        result = [os.path.join(path, f) for f in result]
    if need_sort:
        result.sort()
    return result


def read_to_list(path):
    """
    读取文本文件并将每行存入一个列表

    Parameters:
        path - 文件的路径

    Returns:
        每行文本组成的列表
    """
    try:
        fp = open(path, 'r')
        content = fp.read()
        fp.close()
        return content.split('\n')
    except IOError:
        print("Read from " + path + " failed")


def write_list(out_list, path, type='w'):
    """
    将一个列表按行写到指定文件，如果目录不存在将创建该目录

    Parameters:
        path - 文件的路径

    Returns:
        每行文本组成的列表
    """
    parent_path = '/'.join(path.split('/')[:-1])
    if '/' in path and not os.path.exists(parent_path):
        os.makedirs(parent_path)
    try:
        fp = open(path, type)
        for item in out_list:
            fp.write(str(item) + "\n")
        fp.close()
    except IOError:
        print("Write to " + path + " failed")


def write_str(out_str, path, type='w'):
    """
    将一个字符串写到指定文件，如果目录不存在将创建

    Parameters:
        path - 文件的路径

    Returns:
        每行文本组成的列表
    """
    parent_path = '/'.join(path.split('/')[:-1])
    if '/' in path and not os.path.exists(parent_path):
        os.makedirs(parent_path)
    try:
        fp = open(path, type)
        fp.write(out_str)
        fp.close()
    except IOError:
        print("Write to " + path + " failed")


def export_json(data, path):
    """
    将数据导出到json文件

    Parameters:
        data - 数据
        path - 文件的路径

    Returns:

    """
    parent_path = '/'.join(path.split('/')[:-1])
    if '/' in path and not os.path.exists(parent_path):
        os.makedirs(parent_path)
    f = open(path, 'w')
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
    f.close()


def import_json(path):
    """
        导入json文件

        Parameters:
            path - 文件的路径

        Returns:
            读取的json
    """
    f = open(path)
    return json.load(f)


def get_filename_from_path(path):
    """
        从文件路径中提取文件名

        Parameters:
            path - 文件的路径

        Returns:
            文件名
    """
    if len(path.split('/')) > 1:
        path_arr = path.split('/')
        filename = path_arr[-1]
    else:
        filename = path
    return filename


def un_zip(path):
    """
        解压一个zip文件，注意该方法没有做严格的检查

        Parameters:
            path - 文件的路径

        Returns:
    """
    content = zipfile.ZipFile(path)
    output_dir = path.split('.')[0]
    if os.path.isdir(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    for names in content.namelist():
        content.extract(names, output_dir)
    content.close()


def create_folder(path):
    """
        创建文件目录

        Parameters:
            path - 文件的路径

        Returns:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def export_to_csv(label, label_key, data, data_key, path):
    result = pandas.DataFrame()
    result[label_key] = label
    result[data_key] = data
    result.to_csv(path, index=False)
