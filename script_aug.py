from skimage import morphology, data, color
import cv2 as cv
import numpy as np
import math
import os
import random
import time
import multiprocessing as mp
from Information_extraction import information_extraction
from transformation import flag_judge, identify_reference_corner
from transformation import bezier_transformation, affine_transformation, L2A_transformation


def new_local(src, times=1, stroke_radius=2, k1_control_field_corner=0.6, k2_control_field_third_bezier=0.6, segment=2):

    def deformation(list_all):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
          /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
                    |
        [[cor1, cor2],[ske]]
        """

        # 首先生成需要的数据格式
        list_already, len_list_all, list_pt_reference = [], len(list_all), []
        for a in range(len_list_all):
            list_already.append([[[], []], []])

        # 变形主程序
        for index_list_already in range(len_list_all):
            list_child_already = list_already[index_list_already]
            list_child_all = list_all[index_list_already]
            # 情形判断：几种情况： 1: 1、2空； 2: 1空； 3: 2空； 4: 只有3空
            flag_situation_judge = flag_judge(list_child_already)

            # 每个笔画的变形选择
            if flag_situation_judge == 4:  # 特殊情况4,只能使用贝塞尔曲线变形
                list_already[index_list_already] = bezier_transformation(list_child_already,
                                                                         list_child_all, flag_situation_judge)
                continue  # 特殊情况就不需要后面的扫描过程了
            # 其他都属于正常情况，三种变形方式都能用
            else:
                # 如果是情况1且不是第一对儿角点，需要先指定一个,以最近点作为参考点
                if flag_situation_judge == 1 and index_list_already != 0:
                    flag_situation_judge = 3  # flag标志首先得变，变为flag = 3
                    list_reference = identify_reference_corner(list_all[index_list_already][0][0], list_pt_reference)
                    # index_temp = index_list_already - 1
                    index_temp1, index_temp2 = list_reference[2], list_reference[1]
                    # x_change = list_already[index_temp][0][0][0] - list_all[index_temp][0][0][0]
                    # y_change = list_already[index_temp][0][0][1] - list_all[index_temp][0][0][1]
                    x_change = list_already[index_temp1][0][index_temp2][0] - list_all[index_temp1][0][index_temp2][0]
                    y_change = list_already[index_temp1][0][index_temp2][1] - list_all[index_temp1][0][index_temp2][1]
                    x_get = list_all[index_list_already][0][0][0] + x_change
                    y_get = list_all[index_list_already][0][0][1] + y_change
                    x_already = random.uniform(x_get - list_all[index_list_already][2][0][2][0][0],
                                               x_get + list_all[index_list_already][2][0][2][0][0] + 0.1)
                    y_already = random.uniform(y_get - list_all[index_list_already][2][0][2][0][1],
                                               y_get + list_all[index_list_already][2][0][2][0][1] + 0.1)
                    list_child_already[0][0] = [x_already, y_already]
                    # list_child_already[0][0] = [x_get, y_get]

                # 选择想要的变形方案了 !!!!
                choice = random.randint(1, 1)  # 改这里就可以选择方案，是随机还是指定
                if choice == 1:  # choice为1，选择贝塞尔变形
                    list_already[index_list_already] = bezier_transformation(
                        list_child_already, list_child_all, flag_situation_judge)
                elif choice == 2:  # choice为2，选择Affine变形
                    list_already[index_list_already] = affine_transformation(
                        list_child_already, list_child_all, flag_situation_judge, stroke_radius)
                else:  # choice为3，选择L2A变形
                    list_already[index_list_already] = L2A_transformation(
                        list_child_already, list_child_all, flag_situation_judge, segment, stroke_radius)

            # 扫描相同角点过程,一视同仁
            for index_already_pt in range(2):
                pt_standard = list_all[index_list_already][0][index_already_pt]
                pt_change = list_already[index_list_already][0][index_already_pt]
                list_pt_reference.append((pt_standard, index_already_pt, index_list_already))
                for index_scan in range(index_list_already + 1, len_list_all):
                    for index_scan_child in range(2):
                        if pt_standard == list_all[index_scan][0][index_scan_child]:
                            list_already[index_scan][0][index_scan_child] = pt_change

        return list_already

    def draw_src(list_all):

        # 数据准备
        list_pt, list_judge_x, list_judge_y = [], [], []
        # 先确定字体长宽
        for a in list_all:
            b = a[1]
            for pt in b:
                list_pt.append(pt)
        for a in list_pt:
            list_judge_x.append(a[0]), list_judge_y.append(a[1])
        x_min, x_max = min(list_judge_x), max(list_judge_x)
        y_min, y_max = min(list_judge_y), max(list_judge_y)
        x_len, y_len = x_max - x_min, y_max - y_min

        # 确定底片的长宽
        k1, k2 = 1.1, 1.1
        width, height = int(x_len * k1), int(y_len * k2)

        # 调底片格式
        image_film = np.zeros(shape=(height, width))
        image_film = 255 - image_film
        # print(image_film)
        # exit(0)
        image_film = np.expand_dims(image_film, axis=-1)
        image_film = np.concatenate((image_film, image_film, image_film), axis=-1)

        # 调整点的位置并画图
        size_blank = [int((width - x_len) / 2), int((height - y_len) / 2)]
        for a in list_pt:
            x_new, y_new = a[0] - x_min + size_blank[0], a[1] - y_min + size_blank[1]
            cv.circle(image_film, (x_new, y_new), stroke_radius, (0, 0, 0), -1)
            # cv.circle(image_film, (x_new, y_new), stroke_radius, (255, 255, 255), -1)

        return image_film

    # 信息提取
    list_information = information_extraction(src, k1_control_field_corner, k2_control_field_third_bezier)

    # 变形和画图
    list_draw = []
    while len(list_draw) < times:
        list_final = deformation(list_information)
        picture = draw_src(list_final)
        if picture.shape[2] != 3:
            continue
        list_draw.append(picture)

    return list_draw


if __name__ == '__main__':

    # 单进程的跑法
    # old_time = time.time()
    img = cv.imread("./src/0.jpg", cv.IMREAD_COLOR)
    times = 10
    list_augment = new_local(img, times=times)
    cv.imwrite("./dst/0_src.jpg", img)
    for i in range(len(list_augment)):
        cv.imwrite("./dst/" + str(i + 1) + ".jpg", list_augment[i])
    # print(time.time()-old_time)

    # 多进程的跑法
    # old_time = time.time()
    # img = cv.imread("./pending/124.png", cv.IMREAD_COLOR)
    # cv.imwrite("./After processing/0.jpg", img)
    # p = mp.Pool(os.cpu_count())                               # 创建进程池，池中进程数量为cpu数量
    # list_augment, list_process, result_temp = [], [], []
    # times = 3
    # for i in range(10):                                       # 表示你要处理的进程数量
    #     r = p.apply_async(func=new_local, args=(img, times,))  # 进程打入进程池，如果位置不够，就会让剩余的进程等待；不允许直接在后面加get()，否则会阻塞
    #     list_process.append(r)                                # 这句是为了后面获取返回值，把进程加入到同一list中
    # p.close()                                                 # close必须有，表示进程池不再接受任何除了上述以外的进程,close之后进程池开始运行多进程
    # p.join()                                                  # join必须放在close之后，表示等待进程池所有进程运行结束
    # for res in list_process:                                  # 此处为获取各个进程的返回值
    #     result_temp.append(res.get())                         # get()函数获取各进程返回值并放进列表中
    # # 下面和多进程无关，只是取数据的过程
    # for res in result_temp:
    #     for i in range(len(res)):
    #         list_augment.append(res[i])
    # # 画图
    # for i in range(len(list_augment)):
    #     cv.imwrite("./After processing/" + str(i + 1) + ".jpg", list_augment[i])
    # print(time.time()-old_time)
