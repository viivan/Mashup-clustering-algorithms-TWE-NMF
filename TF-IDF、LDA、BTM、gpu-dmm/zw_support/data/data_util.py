import os

import numpy as np
import pandas as pd
from data import word_divide

"""
csv中获取服务文档信息
预处理，返回
"""


def getDocAsWordArray(filename, fre_num=5):
    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("zw_support\\") + len("zw_support\\")]
    path = os.path.abspath(rootPath + "resource\\" + filename)

    # csv中读取信息
    csv = pd.DataFrame(pd.read_csv(path))
    apiList = csv[["name", "desc", "primary_category"]].loc[0:]
    apiData = np.array(apiList).tolist()

    dic = {}
    name_list = []
    for dataItem in apiData:
        name = dataItem[0]
        name_list.append(name)
        document = dataItem[1]
        dic[name] = document

    dic = word_divide.get_doc_after_divide(dic, fre_num)
    return dic


def getFormerCategory(filename):
    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("zw_support\\") + len("zw_support\\")]
    path = os.path.abspath(rootPath + "resource\\" + filename)

    # num为对应的服务数量
    # csv中读取信息
    csv = pd.DataFrame(pd.read_csv(path))
    apiList = csv[["primary_category"]].loc[0:]
    apiData = np.array(apiList).tolist()

    # 存放标签类型
    c_type = set()
    former = []
    for dataItem in apiData:
        c_type.add(dataItem[0].strip())
    c_type = list(c_type)
    print(set(c_type))
    for dataItem in apiData:
        former.append(c_type.index(dataItem[0].strip()))

    return former


if __name__ == "__main__":
    file_name = "data4.csv"
    dictionary = getDocAsWordArray(file_name)
    cate = getFormerCategory(file_name)
    print(dictionary["Comedify"])
    print(len(cate))
    print(len(set(cate)))
    print(set(cate))
    print(len(dictionary))