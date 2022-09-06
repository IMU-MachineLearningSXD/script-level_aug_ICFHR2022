import numpy as np
import random
import cv2 as cv
import os
import math
from PIL import Image
import time
from skimage import morphology, data, color


# Affine 变形类
class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image # 原始图像
        self.rows = rows # 原始图像的行
        self.cols = cols # 原始图像的列
        self.center = center # 旋转中心，默认是[0,0]

    def Rotate(self,beta):               #旋转
        self.transform_1 = np.array([[math.cos(beta),-math.sin(beta)],
                                     [math.sin(beta), math.cos(beta)]])

    def Zoom(self,factor):               #缩放
        #factor>1表示缩小；factor<1表示放大
        self.transform_2 = np.array([[factor,0],[0,factor]])

    def shear(self, angel):             #沿x错切
        self.transform_3 = np.array([[1,math.tan(angel)],[0,1]])

    def Process_1(self):
        self.dst = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i-self.center[0], j-self.center[1]])
                [x, y] = np.dot(self.transform_1, src_pos)
                [x, y] = np.dot(self.transform_2, [x, y])
                # [x, y, z] = np.dot(self.transform_3, [x, y, z])
                x = int(x)+self.center[0]
                y = int(y)+self.center[1]
                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 0
                else:
                    self.dst[i][j] = self.src[x][y]

    def Process_2(self):
        self.dst = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i-self.center[0], j-self.center[1]])
                [x, y] = np.dot(self.transform_3, src_pos)
                x = int(x)+self.center[0]
                y = int(y)+self.center[1]
                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 0
                else:
                    self.dst[i][j] = self.src[x][y]


# L2A 依赖的类
class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, pt_cor1, pt_cor2, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.pt_cor1 = pt_cor1
        self.pt_cor2 = pt_cor2
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                                 (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)
        pt_cor1_old, pt_cor2_old = (self.pt_cor1[1], self.pt_cor1[0]), (self.pt_cor2[1], self.pt_cor2[0])
        pt_cor1_new, pt_cor2_new = (0, 0), (0, 0)
        flag1, flag2 = 0, 0

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                p, q = i, j
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdx[i, j], self.rdx[i, nj],
                                                 self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdy[i, j], self.rdy[i, nj],
                                                 self.rdy[ni, j], self.rdy[ni, nj])

                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi

                dst[i:i + h, j:j + w] = self.__bilinear_interp(x,
                                                               y,
                                                               self.src[nyi, nxi],
                                                               self.src[nyi, nxi1],
                                                               self.src[nyi1, nxi],
                                                               self.src[nyi1, nxi1]
                                                               )
                for n in range(h):
                    if flag1 == 1 and flag2 == 1:
                        break
                    for m in range(w):
                        if pt_cor1_old in [(nyi[n][m], nxi[n][m]), (nyi[n][m], nxi1[n][m]), (nyi1[n][m], nxi[n][m]),
                                            (nyi1[n][m], nxi1[n][m])]:
                            pt_cor1_new = (q + m, p + n)
                            flag1 = 1
                        if pt_cor2_old in [(nyi[n][m], nxi[n][m]), (nyi[n][m], nxi1[n][m]), (nyi1[n][m], nxi[n][m]),
                                            (nyi1[n][m], nxi1[n][m])]:
                            pt_cor2_new = (q + m, p + n)
                            flag2 = 1

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst, pt_cor1_new, pt_cor2_new


# L2A
def L2A(src, pt_cor1, pt_cor2, segment):

    def distort(src, pt_cor1, pt_cor2, segment):
        img_h, img_w = src.shape[:2]

        cut = img_w // segment
        thresh = cut // 3
        # thresh = img_h // segment // 3
        # thresh = img_h // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
        dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            np.random.randint(thresh) - half_thresh])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            img_h + np.random.randint(thresh) - half_thresh])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h,  pt_cor1, pt_cor2)
        result = trans.generate()

        return result

    def stretch(src, pt_cor1, pt_cor2, segment):
        img_h, img_w = src.shape[:2]

        cut = img_w // segment
        thresh = cut * 4 // 5
        # thresh = img_h // segment // 3
        # thresh = img_h // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h, pt_cor1, pt_cor2)
        result = trans.generate()

        return result

    def perspective(src, pt_cor1, pt_cor2):
        img_h, img_w = src.shape[:2]

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h, pt_cor1, pt_cor2)
        result = trans.generate()

        return result

    k = random.randint(1, 3)
    if k == 1:
        result = distort(src, pt_cor1, pt_cor2, segment)
    elif k == 2:
        result = stretch(src, pt_cor1, pt_cor2, segment)
    else:
        result = perspective(src, pt_cor1, pt_cor2)

    return result


# 骨架索引点
def skeleton_index(skeleton_image):
    index = []
    row, col = skeleton_image.shape
    for a in range(0, row):
        for b in range(0, col):
            if skeleton_image[a][b] == 1:
                index.append((b, a))

    return index


# 求骨架的端点
def endpoint_index(list_ske):
    list_corner = []
    for point in list_ske:
        num_branches = 0
        for a in range(-1, 2):
            for b in range(-1, 2):
                if(point[0] + a, point[1] + b) in list_ske:
                    num_branches = num_branches + 1
        if num_branches == 1 or num_branches == 2:
            list_corner.append(point)

    return list_corner


# 计算两点距离
def two_points_distance(point1, point2):
    p1 = point1[0] - point2[0]
    p2 = point1[1] - point2[1]
    distance = math.hypot(p1, p2)

    return distance


# 角点归位
def pt_cor_back(list_endpoint, pt_cor):
    if not list_endpoint:
        return pt_cor
    else:
        list_distance = []
        for pt in list_endpoint:
            list_distance.append(two_points_distance(pt_cor, pt))
        return list_endpoint[list_distance.index(min(list_distance))]


# flag判定
def flag_judge(list_child_already):

    if not list_child_already[0][0]:
        if not list_child_already[0][1]:
            flag_situation_judge = 1  # 角点1空，角点2也空
        else:
            flag_situation_judge = 2  # 角点1空，角点2不空
    elif not list_child_already[0][1]:
        flag_situation_judge = 3  # 角点1不空，角点2空
    else:
        flag_situation_judge = 4  # 角点1不空，角点2也不空

    return flag_situation_judge


# 指定一个角点
def identify_reference_corner(pt_wait, list_reference):

    list_distance = []
    for list_child in list_reference:
        pt_compare = list_child[0]
        list_distance.append(two_points_distance(pt_wait, pt_compare))
    index_min = list_distance.index(min(list_distance))
    list_need = list_reference[index_min]

    return list_need


# 方法一：贝塞尔曲线变形
def bezier_transformation(list_already, list_information, flag_special):
    """
        list_information:
        [[list_cor], [list_ske], [list_bezier_use_information]]
            /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
        [[pt_cor1, pt_cor2],[pt_third_bezier]]     # 这是list_total 需要的格式
                    |
        [[cor1, cor2],[ske]]     # 这是list_already 需要的格式
    """

    # 拿数据
    list_bezier_inf = list_information[2]  # 取整个信息中贝塞尔变形所用部分
    list_total = []
    for k in range(len(list_bezier_inf)):
        list_total.append([[[], []], []])
    # 根据list_already的不同情形先把list_total的角点打上去
    if flag_special == 2:  # 角点1空，角点2不空
        list_total[-1][0][1] = list_already[0][1]
    elif flag_special == 3:  # 角点1不空，角点2空
        list_total[0][0][0] = list_already[0][0]
    elif flag_special == 4:  # 角点1不空，角点2也不空
        list_total[-1][0][1] = list_already[0][1]
        list_total[0][0][0] = list_already[0][0]

    # 以list_total为主循环，算出数据并打入,算list_total
    for index_list_total in range(len(list_total)):
        list_total_cor = list_total[index_list_total][0]
        list_bezier = list_bezier_inf[index_list_total]
        # 先处理第一个角点
        if not list_total[index_list_total][0][0]:  # 表示角点1是空的,只有空的才处理
            # 表示角点2也是空的，两个都空,说明是整个图像的第一对儿,则直接以自己为基准确定落点
            if not list_total[index_list_total][0][1]:
                x_get, y_get = list_bezier[0][0][0], list_bezier[0][0][1]
                x_already = random.uniform(x_get - list_bezier[2][0][0], x_get + list_bezier[2][0][0] + 0.1)
                y_already = random.uniform(y_get - list_bezier[2][0][1], y_get + list_bezier[2][0][1] + 0.1)
                pt_already = [x_already, y_already]
                list_total[index_list_total][0][0].extend(pt_already)
            # 表示角点2存在，所以要以角点2为基准点
            else:
                x_change = list_total_cor[1][0] - list_bezier[0][1][0]
                y_change = list_total_cor[1][1] - list_bezier[0][1][1]
                x_get, y_get = list_bezier[0][0][0] + x_change, list_bezier[0][0][1] + y_change
                x_already = random.uniform(x_get - list_bezier[2][0][0], x_get + list_bezier[2][0][0] + 0.1)
                y_already = random.uniform(y_get - list_bezier[2][0][1], y_get + list_bezier[2][0][1] + 0.1)
                pt_already = [x_already, y_already]
                list_total[index_list_total][0][0].extend(pt_already)
        # 再处理第二个角点
        if not list_total[index_list_total][0][1]:  # 表示角点2是空的，只有空的才处理，此时，角点1一定存在，所以以角点1为基准点确定落点
            x_change = list_total_cor[0][0] - list_bezier[0][0][0]
            y_change = list_total_cor[0][1] - list_bezier[0][0][1]
            x_get, y_get = list_bezier[0][1][0] + x_change, list_bezier[0][1][1] + y_change
            x_already = random.uniform(x_get - list_bezier[2][1][0], x_get + list_bezier[2][1][0] + 0.1)
            y_already = random.uniform(y_get - list_bezier[2][1][1], y_get + list_bezier[2][1][1] + 0.1)
            pt_already = [x_already, y_already]
            list_total[index_list_total][0][1].extend(pt_already)
            # 如果还有下一对，就需要把这个点补到第一个点
            if index_list_total + 1 < len(list_total):
                list_total[index_list_total + 1][0][0].extend(pt_already)
        # 有了两个角点后，计算第三贝塞尔控制点
        pt_cor1_new, pt_cor2_new = list_total_cor[0], list_total_cor[1]
        pt_cor1_old, pt_cor2_old = list_bezier[0][0], list_bezier[0][1]
        x_corner1_change, y_corner1_change = pt_cor1_new[0] - float(pt_cor1_old[0]), pt_cor1_new[1] - float(pt_cor1_old[1])
        x_corner2_change, y_corner2_change = pt_cor2_new[0] - float(pt_cor2_old[0]), pt_cor2_new[1] - float(pt_cor2_old[1])
        x_change = float(0.5 * x_corner1_change) + float(0.5 * x_corner2_change)
        y_change = float(0.5 * y_corner1_change) + float(0.5 * y_corner2_change)
        x_get, y_get = list_bezier[1][0][0] + x_change, list_bezier[1][0][1] + y_change
        x_already = random.uniform(x_get - list_bezier[3][0], x_get + list_bezier[3][0] + 0.1)
        y_already = random.uniform(y_get - list_bezier[3][0], y_get + list_bezier[3][0] + 0.1)
        pt_already = [x_already, y_already]
        list_total[index_list_total][1].extend(pt_already)

    # list_total所有处理完之后，就把list_total[0][0][0] 和 list_total[-1][0][1] 替换 list_already[0]就行
    list_already[0] = [list_total[0][0][0],  list_total[-1][0][1]]
    # 然后再求list_already里的ske
    list_ske = []

    # 做list_already
    for list_total_child in list_total:
        pt_cor1, pt_cor2 = np.array(list_total_child[0][0]), np.array(list_total_child[0][1]),
        pt_bezier = np.array(list_total_child[1])
        p = lambda t: (1 - t) ** 2 * pt_cor1 + 2 * t * (1 - t) * pt_bezier + t ** 2 * pt_cor2
        points = np.array([p(t) for t in np.linspace(0, 1, 500)])
        x, y = points[:, 0].astype(np.int).tolist(), points[:, 1].astype(np.int).tolist()
        # 一个个点打进list_ske
        for t in range(len(x)):
            list_ske.append((x[t], y[t]))
    # 去一次重复的点
    list_ske = list(set(list_ske))
    # 数据打入list_already
    list_already[1].extend(list_ske)

    return list_already


# 方法二：Affine
def affine_transformation(list_already, list_information, flag_special, stroke_radius):
    """
    list_information:
        [[list_cor], [list_ske], [list_bezier_use_information]]
            /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
        [[cor1, cor2],[ske]]     # 这是list_already 需要的格式
    """

    list_cor, list_ske = list_information[0], list_information[1]
    list_all_pt, list_judge_x, list_judge_y = [], [], []
    list_all_pt.extend(list_ske)
    for pt in list_all_pt:
        list_judge_x.append(pt[0]), list_judge_y.append(pt[1])
    x_min, x_max = min(list_judge_x), max(list_judge_x)
    y_min, y_max = min(list_judge_y), max(list_judge_y)
    x_len, y_len = x_max - x_min, y_max - y_min

    # 参数
    parm_zoom = random.uniform(0.7, 1.3)
    parm_rotate = math.radians(random.randint(-10, 10))
    parm_shear = math.radians(random.randint(-10, -10))   # tan限制要在（-90，90）

    # 确定底片的大小
    # k = 3    # 之前一视同仁的变形区域
    # shape = int(max(x_len, y_len) * k)
    shape = int(math.sqrt(2) * max(x_len, y_len) * float(1/parm_zoom))
    if shape < 30:
        shape += 20

    # 调底片格式
    image_film = Image.new("RGB", (shape, shape), "black")
    image_film = np.array(image_film)
    # 调整点的位置并画图
    size_blank = [int((shape - x_len) / 2), int((shape - y_len) / 2)]
    for a in list_all_pt:
        x_new, y_new = a[0] - x_min + size_blank[0], a[1] - y_min + size_blank[1]
        cv.circle(image_film, (x_new, y_new), stroke_radius, (255, 255, 255), -1)
    # 点的位置
    pt_cor1 = (list_cor[0][0] - x_min + size_blank[0], list_cor[0][1] - y_min + size_blank[1])
    pt_cor2 = (list_cor[1][0] - x_min + size_blank[0], list_cor[1][1] - y_min + size_blank[1])
    src = cv.cvtColor(image_film, cv.COLOR_BGR2GRAY)
    rows = src.shape[0]
    cols = src.shape[1]
    center = [int(rows / 2), int(cols / 2)]

    # 变形
    # 算点最终落在哪里
    rotate = np.array([[math.cos(parm_rotate), -math.sin(parm_rotate)], [math.sin(parm_rotate), math.cos(parm_rotate)]])
    zoom = np.linalg.inv(np.array([[parm_zoom, 0], [0, parm_zoom]]))
    shear = np.array([[1, 0], [-math.tan(parm_shear), 1]])

    # 点1
    src_pos = np.array([pt_cor1[0] - center[0], pt_cor1[1] - center[1]])
    [x, y] = np.dot(rotate, src_pos)
    [x, y] = np.dot(zoom, [x, y])
    [x, y] = np.dot(shear, [x, y])
    cor1 = (int(x) + center[0], int(y) + center[1])
    # 点2
    src_pos = np.array([pt_cor2[0] - center[0], pt_cor2[1] - center[1]])
    [x, y] = np.dot(rotate, src_pos)
    [x, y] = np.dot(zoom, [x, y])
    [x, y] = np.dot(shear, [x, y])
    cor2 = (int(x) + center[0], int(y) + center[1])

    # 处理图形，做旋转、缩放、错切变换
    img = Img(src, rows, cols, center)
    img.Rotate(parm_rotate)  # 旋转
    img.Zoom(parm_zoom)      # 缩放
    img.Process_1()
    img1 = Img(img.dst, rows, cols, center)
    img1.shear(parm_shear)   # 错切
    img1.Process_2()

    # 变形后的笔画图模糊一次（去毛刺）
    stroke_affine = 255 - img1.dst
    stroke_affine = (stroke_affine < 127)
    stroke_affine = stroke_affine.astype(np.uint8) * 255
    stroke_affine = np.expand_dims(stroke_affine, axis=-1)
    stroke_affine = np.concatenate((stroke_affine, stroke_affine, stroke_affine), axis=-1)

    # test 要删掉
    # cv.imshow('kk',stroke_affine)
    # cv.waitKey(0)
    # exit(0)

    img_blur = cv.blur(stroke_affine, (3, 3))
    img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    img_blur = (img_blur > 127)
    # 模糊后做骨架检测
    img_ske = morphology.skeletonize(img_blur)

    # 新的骨架list
    list_ske_new = skeleton_index(img_ske)

    # 角点归位
    list_endpoint = endpoint_index(list_ske_new)
    cor1 = pt_cor_back(list_endpoint, cor1)
    cor2 = pt_cor_back(list_endpoint, cor2)

    # 计算新的点（就是一次平移）
    x_mark, y_mark = 0, 0
    if flag_special == 3:
        x_mark, y_mark = list_already[0][0][0] - cor1[0], list_already[0][0][1] - cor1[1]
        list_already[0][1] = [int(cor2[0] + x_mark), int(cor2[1] + y_mark)]
    elif flag_special == 2:
        x_mark, y_mark = list_already[0][1][0] - cor2[0], list_already[0][1][1] - cor2[1]
        list_already[0][0] = [int(cor1[0] + x_mark), int(cor1[1] + y_mark)]
    elif flag_special == 1:
        list_already[0][0], list_already[0][1] = [int(cor1[0]), int(cor1[1])], [int(cor2[0]), int(cor2[1])]
    for pt in list_ske_new:
        list_already[1].append([int(pt[0] + x_mark), int(pt[1] + y_mark)])

    return list_already


# 方法一：L2A变形
def L2A_transformation(list_already, list_information, flag_special, segment, stroke_radius):
    """
        list_information:
            [[list_cor], [list_ske], [list_bezier_use_information]]
                /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                        |
            [[cor1, cor2],[ske]]     # 这是list_already 需要的格式
        """
    list_cor, list_ske = list_information[0], list_information[1]
    list_all_pt, list_judge_x, list_judge_y = [], [], []
    list_all_pt.extend(list_ske)
    for pt in list_all_pt:
        list_judge_x.append(pt[0]), list_judge_y.append(pt[1])
    x_min, x_max = min(list_judge_x), max(list_judge_x)
    y_min, y_max = min(list_judge_y), max(list_judge_y)
    x_len, y_len = x_max - x_min, y_max - y_min

    # 确定底片的大小
    k = 1.5    # 之前一视同仁的变形区域
    shape = int(max(x_len, y_len) * k)
    if shape < 30:
        shape = 35

    # 调底片格式
    image_film = Image.new("RGB", (shape, shape), "black")
    image_film = np.array(image_film)
    # 调整点的位置并画图
    size_blank = [int((shape - x_len) / 2), int((shape - y_len) / 2)]
    for a in list_all_pt:
        x_new, y_new = a[0] - x_min + size_blank[0], a[1] - y_min + size_blank[1]
        cv.circle(image_film, (x_new, y_new), stroke_radius, (255, 255, 255), -1)

    # 点的位置
    pt_cor1 = (list_cor[0][0] - x_min + size_blank[0], list_cor[0][1] - y_min + size_blank[1])
    pt_cor2 = (list_cor[1][0] - x_min + size_blank[0], list_cor[1][1] - y_min + size_blank[1])

    # 变形
    src = cv.cvtColor(image_film, cv.COLOR_BGR2GRAY)
    result = L2A(src, pt_cor1, pt_cor2, segment)
    dst, cor1, cor2 = result[0], result[1], result[2]
    stroke_l2a = 255 - dst
    stroke_l2a = (stroke_l2a < 127)
    stroke_l2a = stroke_l2a.astype(np.uint8) * 255
    stroke_l2a = np.expand_dims(stroke_l2a, axis=-1)
    stroke_l2a = np.concatenate((stroke_l2a, stroke_l2a, stroke_l2a), axis=-1)
    img_blur = cv.blur(stroke_l2a, (3, 3))
    img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    img_blur = (img_blur > 127)
    # 模糊后做骨架检测
    img_ske = morphology.skeletonize(img_blur)

    # 新的骨架list
    list_ske_new = skeleton_index(img_ske)

    # 角点归位
    list_endpoint = endpoint_index(list_ske_new)
    cor1 = pt_cor_back(list_endpoint, cor1)
    cor2 = pt_cor_back(list_endpoint, cor2)

    # 计算新的点（就是一次平移）
    x_mark, y_mark = 0, 0
    if flag_special == 3:
        x_mark, y_mark = list_already[0][0][0] - cor1[0], list_already[0][0][1] - cor1[1]
        list_already[0][1] = [int(cor2[0] + x_mark), int(cor2[1] + y_mark)]
    elif flag_special == 2:
        x_mark, y_mark = list_already[0][1][0] - cor2[0], list_already[0][1][1] - cor2[1]
        list_already[0][0] = [int(cor1[0] + x_mark), int(cor1[1] + y_mark)]
    elif flag_special == 1:
        list_already[0][0], list_already[0][1] = [int(cor1[0]), int(cor1[1])], [int(cor2[0]), int(cor2[1])]
    for pt in list_ske_new:
        list_already[1].append([int(pt[0] + x_mark), int(pt[1] + y_mark)])

    return list_already



