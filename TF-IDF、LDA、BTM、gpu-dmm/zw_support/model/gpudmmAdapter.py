import os

import data.dimension_reduce as dr
import visible.coordinatepainting as cp

"""
主要是读取文件
有机会用python重写下训练过程
https://github.com/NobodyWHU/GPUDMM 参考
"""


def loadModel(save_file):
    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("zw_support\\") + len("zw_support\\")]
    path = os.path.abspath(rootPath + "result\\" + save_file)

    # 读取文档分布矩阵
    try:
        pf = open(path, "r+")
        result = []  # 存放文档分布
        lines = pf.readlines()
        for line in lines:
            line = line.strip()
            nums = line.split(" ")
            result.append([float(i) for i in nums])
        pf.close()
        return result
    except IOError:
        print("文件无法正常打开")
        return None


if __name__ == "__main__":
    save_file_name = "12_gpudmm_doc_topic.txt"
    doc_topic = loadModel(save_file_name)

    title = "GPU-DMM"
    print("正在降维")
    d2_data = dr.dimension_down(doc_topic)
    cp.paintModelPoint(d2_data, title)
