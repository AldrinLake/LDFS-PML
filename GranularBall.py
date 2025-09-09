
"""
granular ball
"""
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.cluster import k_means
import warnings
import random
import copy
warnings.filterwarnings("ignore")


class GranularBall:
    """ class of the granular ball
        data format: [{attribute1,attribute2,...,attributeN}, {label}, {index}]
    """
    def __init__(self, data, data_all):
        """
        :param data: a subuniverse used to generated ball. Labeled data set, the "-2" column is the class label, the last column is the index of each line
        :param data_all: whole universe
        and each of the preceding columns corresponds to a feature
        """
        self.data_all = data_all # whole universe
        self.data = data[:, :]  # subuniverse for generating the ball
        self.data_no_label = data[:, :-2]
        self.center = self.data_no_label.mean(0)    # compress over rows, compute the mean of each column
        self.radius = self.__get_radius()
        self.cover_data = self.__get_cover_data()
        self.num, self.dim = self.cover_data[:, :-2].shape
        self.purity = self.__get_purity()   # get the current granular ball purity
        if self.purity >0.5:
            self.label=1
        else:
            self.label=0
        # self.label = np.round(self.purity)


    def __get_purity(self):
        """
        :return: the purity of the granular ball.
        """
        labels_in_ball = self.cover_data[:, -2]
        # label value 2 denotes pseudo-positive; convert 2 to 1 for purity calculation
        purity = np.sum(np.where(labels_in_ball==2,1,labels_in_ball)) / self.num
        return purity

    def __get_cover_data(self):
        """
        : return: the object covered by the ball with its radius ： formalized as O in the paper
        """
        dis_matrix = cdist([self.center], self.data_all[:, :-2], metric='euclidean')
        neighbor_point_index = np.where(dis_matrix <= self.radius)[1]  # indicate the column index in distance matrix
        return self.data_all[neighbor_point_index]



    def __get_radius(self):
        radius = np.max((((self.data_no_label - self.center) ** 2).sum(axis=1) ** 0.5))
        return radius

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        # set the same random number seed to ensure the stability of each clustering result
        label_cluster = k_means(X=self.data_no_label, n_clusters=2, random_state=123)[1]
        # label_cluster = k_means(X=self.data_no_label, n_clusters=2)[1]

        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :], self.data_all)
            ball2 = GranularBall(self.data[label_cluster == 1, :], self.data_all)
        else:
            ball1 = GranularBall(self.data[0:1, :], self.data_all)
            ball2 = GranularBall(self.data[1:, :], self.data_all)
        return ball1, ball2


class GBList:
    """ class of the list of granular ball
        data format: [{attribute1,attribute2,...,attributeN}, {label}, {index}]
    """
    def __init__(self, data=None):
        self.data_temp = data[:, :]
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data, self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, purity=1.0, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ball_generate_record = []
        ll = len(self.granular_balls)
        i = 0
        while True:
            # print("purity:{}, radius:{},size:{},c_data:{}".format(self.granular_balls[i].purity,self.granular_balls[i].radius,len(self.granular_balls[i].data),len(self.granular_balls[i].cover_data)))
            if self.granular_balls[i].purity < purity and self.granular_balls[i].purity>0:
                if len(self.granular_balls[i].data)==1 and len(self.granular_balls[i].cover_data)>1:
                    # In this case, there exist duplicate data with different labels but the same attributes
                    i += 1
                    if i >= ll:
                        break
                    else:
                        continue
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break

        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.cover_data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_radius(self):
        return list(map(lambda x: x.radius, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)  # stack into row-wise matrix

    def get_data_covered_by_ball_that_size_large_than_k(self, k:int):
        """
        :return: Data covered by balls' radius; the size of these ball is large then k
        """
        # print(k)
        list_data = [ball.data for ball in self.granular_balls if ball.num >= k]
        return np.vstack(list_data)  # stack into row-wise matrix


    def remove_negative_balls(self):
        self.granular_balls = [ball for ball in self.granular_balls if ball.label == 1]
        self.data = self.get_data()
        return self.granular_balls

    def remove_void_balls(self):
        self.granular_balls = [ball for ball in self.granular_balls if  ball.num >= 1]




    def merge_two_nearest_ball(self):
        """
        the larger ball A will merge the smaller one B if satisfies following conditions:
        1. the distance between the centers of A and B is smaller than the sum of their radius
        2. the merge of two balls A and B will not reduce the purity of the original one
        """
        while True:
            center_list = self.get_center()
            radius_list = self.get_radius()
            size_list = self.get_data_size()
            flag = False  # determine whether there is a ball merge
            for i in range(len(self.granular_balls)):
                for j in range(len(self.granular_balls)):
                    if i == j :
                        continue
                    # merge two balls that size>1
                    if size_list[i]>size_list[j] and np.linalg.norm(np.array(center_list[i]) - np.array(center_list[j])) < (radius_list[i]+radius_list[j]):
                        new_ball_temp = GranularBall(data=np.vstack((self.granular_balls[i].data, self.granular_balls[j].data)), data_all=self.data_temp)

                        if new_ball_temp.purity <= 0.99999999:
                            continue
                        else:
                            # print(new_ball_purity)
                            temp_ball1 = self.granular_balls[i]
                            temp_ball2 = self.granular_balls[j]
                            self.granular_balls.remove(temp_ball1)
                            self.granular_balls.remove(temp_ball2)
                            self.granular_balls.append(new_ball_temp)
                            flag = True
                            break
                    # merge two balls that size=1
                    # if size_list[i]==1 and size_list[j]==1:
                    #     new_ball_temp = GranularBall(data=np.vstack((self.granular_balls[i].data, self.granular_balls[j].data)),data_all=self.data_temp)
                    #     new_ball_temp_purity = new_ball_temp.purity
                    #     if new_ball_temp_purity <= 0.99999999:
                    #         continue
                    #     else:
                    #         # print(new_ball_purity)
                    #         temp_ball1 = self.granular_balls[i]
                    #         temp_ball2 = self.granular_balls[j]
                    #         self.granular_balls.remove(temp_ball1)
                    #         self.granular_balls.remove(temp_ball2)
                    #         self.granular_balls.append(new_ball_temp)
                    #         flag = True
                    #         break
                    # merge two balls that size(ball1)>1 and size(ball2)=1
                    # if size_list[i]>1 and size_list[j]==1:
                    #     new_ball_temp = GranularBall(data=np.vstack((self.granular_balls[i].data, self.granular_balls[j].data)),data_all=self.data_temp)
                    #     new_ball_temp_purity = new_ball_temp.purity
                    #     if new_ball_temp_purity <= 0.99999999:
                    #         continue
                    #     else:
                    #         # print(new_ball_purity)
                    #         temp_ball1 = self.granular_balls[i]
                    #         temp_ball2 = self.granular_balls[j]
                    #         self.granular_balls.remove(temp_ball1)
                    #         self.granular_balls.remove(temp_ball2)
                    #         self.granular_balls.append(new_ball_temp)
                    #         flag = True
                    #         break
                if flag:
                    break
            if flag == False:
                break
        return self.granular_balls

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        label_cluster = k_means(X=self.data[:, :-2], n_clusters=k, init=self.get_center())[1]
        for i in range(k):
            self.granular_balls[i] = GranularBall(self.data[label_cluster == i, :])

    def re_division(self, i):
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        label_cluster = k_means(X=self.data[:, attributes], n_clusters=k,
                                init=self.get_center()[:, attributes], max_iter=1)[1]
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division