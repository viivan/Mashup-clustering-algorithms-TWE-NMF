import matplotlib.pyplot as plt
import random
import numpy as np

color = ["om", "og", "or", "oy", "ok", "ob", "oc", "*m", "*g", "*r","*y","*k"]
text_color = ["purple", "green", "red", "goldenrod", "black", "blue", "sienna", "navy","violet", "grey", "brown", "pink"]
line_color = ["m-", "g-", "r-", "y-", "k-", "b-", "c-"]


def paintClusterResult(data, label, under_label):
    # 输出对应的纯度，RI，F1，熵的聚类准确度结果矢量图
    # 预计为条形图,data为[[]], 存放对应数据
    x = range(len(data[0]))
    x = [i * 2 for i in x]
    width = 0.26  # 方体宽度
    for i in range(len(data)):
        d = data[i]
        x_place = [p + width * i for p in x]  # 每个方体的x坐标
        rect = plt.bar(x=x_place, height=d, width=width, alpha=0.6, color=text_color[i % len(text_color)], label=label[i])
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2, height+0.1, str(round(height, 2)), ha="center", va="bottom")
    plt.ylabel("Num")
    plt.ylim(0, 4)
    plt.xticks([i + width / 2 * (len(data) - 1) for i in x], under_label)
    plt.xlabel("Method")
    plt.title("Clustering Validation Method")
    plt.legend()
    plt.show()


def paintLineChart(method_label, title, result, under_label):
    """
    绘制某个数值不同的方法的折线图
    result计划为对应的不同情况下准确值list，一种method一个list
    method_label为对应方法名，under_label为x轴提示
    """
    x = range(len(under_label))
    x = [i * 3 for i in x]
    t = 0
    for r in result:
        plt.plot(x, r, line_color[t % len(line_color)], linewidth=1, label=method_label[t])
        for i in range(len(r)):
            # 绘制点，更显眼
            plt.plot(x[i], r[i], color[t % len(color)], ms=5)
        t += 1

    plt.title(title)
    plt.ylabel("Num")
    plt.ylim(0, 4)
    plt.xticks(x, under_label)

    plt.legend()
    plt.show()


def printClusterByPointInD2(data,result, title, num=5):
    """
    主要将降维处理后的数据进行可视化
    label为对应点的名称，看情况输出
    result为聚类结果，title为图标题
    二维,num为每个类数量
    """
    t = 0
    f_size = 5
    n_count = [0] * len(set(result))  # 计算每个类显示数量
    #print(result)
    for p in data:
        if num >= 0:
            if n_count[result[t]] >= num:
                t += 1
                continue
            else:
                font = {
                    'weight': 'normal',
                    'color': text_color[result[t] % len(text_color)],
                    'size': f_size + 2
                }
                n_count[result[t]] += 1
                plt.plot(p[0] * f_size, p[1] * f_size, color=color[result[t] % len(color)], ms=5)
                # plt.text(p[0] * f_size, p[1] * f_size, label[t], fontdict=font)
                t += 1
        else:
            font = {
                'weight': 'normal',
                'color': text_color[result[t] % len(text_color)],
                'size': f_size + 2
            }
            n_count[result[t]] += 1
            plt.plot(p[0] * f_size, p[1] * f_size, color[result[t] % len(color)], ms=5)
            t += 1
    plt.title(title)
    plt.show()


def printTopicWordByPointInD2(word_vec, word, title):
    """
    将同一类的topic——word的词向量降维后可视化
    不同类点颜色更改
    每一类最多num个词
    同时输出word_text
    """
    t = 0
    f_size = 12
    count = 0;
    #n_count = [0] * len(set(result))  # 计算每个类显示数量
    #photo，
    for i in (0,4,8,7,13):
    #for i in range(12):
        j = 0;
        for vec in word_vec[i]:
            font = {
                'weight': 'normal',
                'color': text_color[count],
                'size': f_size
             }
            plt.plot(vec[0], vec[1], text_color[count], ms=3, marker="o")
            plt.text(vec[0], vec[1], word[i][j], fontdict=font)
            j+=1;
        count+=1

    #plt.title(title)
    plt.show()


if __name__ == "__main__":
    data = [[1.0, 1.0], [0.50287264585495, 0.0], [0.0, 0.9961761236190796]]
    label = ["type1", "type2", "type3"]
    result = [0, 1, 1]
    title = "test"
    printClusterByPointInD2(data, label, result, title)