import matplotlib.pyplot as plt
import numpy as np

color = ["or", "og", "om", "oy", "ok", "ob", "oc", "*m", "*g", "*r"]
text_color = ["red", "green", "purple", "goldenrod", "black", "blue", "sienna", "navy", "violet", "grey"]
line_color = ["r-", "g-", "m-", "y-", "k-", "b-", "c-"]


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
    plt.ylim(np.min(result) - 5, np.max(result) + 5)
    plt.xticks(x, under_label)

    plt.legend()
    plt.show()


def printClusterByPointInD2(data, label, result, title, num=5):
    """
    主要将降维处理后的数据进行可视化
    label为对应点的名称，看情况输出
    result为聚类结果，title为图标题
    二维,num为每个类数量
    """
    t = 0
    f_size = 5
    n_count = [0] * len(set(result))  # 计算每个类显示数量
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
                plt.text(p[0] * f_size, p[1] * f_size, label[t], fontdict=font)
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


def printTopicWordByPointInD2(word_vec, word, result, title, num=5):
    """
    将同一类的topic——word的词向量降维后可视化
    不同类点颜色更改
    每一类最多num个词
    同时输出word_text
    """
    t = 0
    f_size = 13
    n_count = [0] * len(set(result))  # 计算每个类显示数量
    for vec in word_vec:
        if n_count[result[t]] >= num:
            t += 1
            continue
        else:
            font = {
                'weight': 'normal',
                'color': text_color[result[t] % len(text_color)],
                'size': f_size
            }
            n_count[result[t]] += 1
            plt.plot(vec[0], vec[1], text_color[result[t] % len(text_color)], ms=5, marker="o")
            plt.text(vec[0], vec[1], word[t], fontdict=font)
            t += 1

    plt.title(title)
    plt.show()


def paintModelPoint(data, title):
    """
    主要将降维处理后的数据进行可视化
    这个方法用同一颜色输出所有的点
    """
    t = 0
    f_size = 5
    for p in data:
        plt.plot(p[0] * f_size, p[1] * f_size, color[5], ms=5)
        t += 1
    plt.title(title)
    plt.show()


def paintWordIntoPoint(word_vec, words, title):
    f_size = 13
    for i in range(len(word_vec)):
        font = {
            'weight': 'normal',
            'color': text_color[i],
            'size': f_size
        }
        for j in range(len(word_vec[i])):
            plt.plot(word_vec[i][j][0], word_vec[i][j][1], text_color[i], ms=5, marker="o")
            plt.text(word_vec[i][j][0], word_vec[i][j][1], words[i][j], fontdict=font)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    r_words = [['price', 'product', 'find', 'online', 'deal'], ['phone', 'message', 'call', 'text', 'using'], ['search', 'google', 'result', 'api', 'twitter'], ['music', 'artist', 'video', 'new', 'track'], ['travel', 'map', 'world', 'information', 'destination']]
    r_word_vec = [[[0.8424984216690063, 0.7871577739715576], [0.48387935757637024, 0.45157358050346375], [0.5556740164756775, 0.6542916297912598], [0.5007752180099487, 0.5900745391845703], [0.5220575332641602, 0.5176807045936584]], [[0.7973381876945496, 0.32767149806022644], [0.7681403756141663, 0.4166184663772583], [0.7096080183982849, 0.3524153232574463], [0.6802012920379639, 0.43370985984802246], [0.6113079786300659, 0.7507913708686829]], [[0.20295393466949463, 0.02235543355345726], [0.27100566029548645, 0.1999916285276413], [0.3027631938457489, 0.7264106273651123], [0.3821564316749573, 0.6945729851722717], [0.41508379578590393, 0.6252418756484985]], [[0.42692241072654724, 0.15474949777126312], [0.4890446066856384, 0.12975642085075378], [0.07458721846342087, 0.3109928071498871], [0.39712613821029663, 0.28916463255882263], [0.45487767457962036, 0.34135186672210693]], [[0.898629367351532, 0.5365327596664429], [0.9935160279273987, 0.5972675681114197], [0.7758117318153381, 0.5760762095451355], [0.7469831109046936, 0.6607539653778076], [0.7029008269309998, 0.5416252613067627]]]
    paintWordIntoPoint(r_word_vec, r_words, " ")