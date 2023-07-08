import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math

class Vocabulary:
    def __init__(self, k):
        self.k = k
        self.vocabulary = None

    def generateBoW(self, path, random_state):
        """

        通过聚类算法，生成词袋模型

        :param path:词袋模型存储路径

        :param self.k: 词袋模型中视觉词汇的个数

        :param random_state: 随机数种子

        :return: 词袋模型视觉词汇矩阵

        """
        featureSet = np.load(path)
        np.random.shuffle(featureSet)
        print(f"sift features number: {featureSet.shape}")
        kmeans = MiniBatchKMeans(n_clusters=self.k,random_state=random_state,batch_size=200).fit(featureSet)
        centers = kmeans.cluster_centers_
        self.vocabulary = centers

        np.save("BOW.npy",centers)
        return centers

    def getBow(self, path):
        """

        读取词袋模型文件

        :param path: 词袋模型文件路径

        :return: 词袋模型矩阵

        """
        centers = np.load(path)
        self.vocabulary = centers
        return centers

    def calSPMFeature(self, features, keypoints, center, img_x, img_y, numberOfBag):
        '''
        使用 SPM 算法，生成不同尺度下图片对视觉词汇的投票结果向量
        :param features:图片的特征点向量 1x128
        :param keypoints: 图片的特征点列表 nx2
        :param center: 词袋中的视觉词汇的向量 k*128
        :param img_x: 图片的宽度
        :param img_y: 图片的长度
        :param self.k: 词袋中视觉词汇的个数
        :return: 基于 SPM 思想生成的图片视觉词汇投票结果向量

        '''
        size = len(features)
        # ----------------------write your code bellow----------------------

        # 4x4 block
        widthStep = math.ceil(img_x / 4)
        heightStep = math.ceil(img_y / 4)

        histogramOfLevelZero = np.zeros((1, numberOfBag))
        histogramOfLevelOne = np.zeros((4, numberOfBag))
        histogramOfLevelTwo = np.zeros((16, numberOfBag))

        for i in range(size): # see all the features

            feature = features[i]   
            keypoint = keypoints[i]
            x, y = keypoint.pt

            boundaryIndex = math.floor(x / widthStep) + math.floor(y / heightStep) * 4 # idx of the img block

            diff = np.tile(feature, (numberOfBag, 1)) - center # diff:k*128
            SquareSum = np.sum(np.square(diff), axis=1) # k*1

            index = np.argmin(SquareSum) # find most similar feature's index
            histogramOfLevelTwo[boundaryIndex][index] += 1

        # fix by joehuang
        origin_list = [0,2,8,10]
        for i in range(4):
                origin = origin_list[i]
                histogramOfLevelOne[i] = histogramOfLevelTwo[origin] + histogramOfLevelTwo[origin + 1] + histogramOfLevelTwo[origin + 4] + histogramOfLevelTwo[origin + 5]

        for i in range(4):
            histogramOfLevelZero[0] += histogramOfLevelOne[i]
        
        result = np.float32([]).reshape(0, numberOfBag) # X*K
        result = np.append(result, histogramOfLevelZero * 0.25, axis=0)
        result = np.append(result, histogramOfLevelOne * 0.25, axis=0)
        result = np.append(result, histogramOfLevelTwo * 0.5, axis=0) # 21*K
        # ----------------------write your code above----------------------
        return result

    def Imginfo2SVMdata(self, data):
        """

        将图片特征点数据转化为 SVM 训练的投票向量
        convert images's sift features to a total representation

        :param self.vocabulary: 词袋模型

        :param datas: 图片特征点数据

        :param self.k:

        词袋模型中视觉词汇的数量

        :return: 投票向量矩阵，图片标签

        """
        dataset = np.float32([]).reshape(0, self.k * 21)
        labels = []
        # ----------------------write your code bellow----------------------
        for simple in data:
            votes = self.calSPMFeature(simple.descriptors
            , simple.keypoints, self.vocabulary, simple.width, simple.height, self.k)
            # 21 means 1 + 4x1 + 4x4 这里展平了 
            votes = votes.ravel().reshape(1, self.k * 21)
            dataset = np.append(dataset, votes, axis=0)
            labels.append(simple.label)
        # ----------------------write your code above----------------------
        labels = np.array(labels)

        return dataset, labels

