import numpy as np

class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.forword_p = None # 前向概率
        self.beats = None
        self.backward_p = None

    def forward(self,Q,V,A,B,O,PI):
        """
        :param Q: 所有可能的隐藏状态的集合
        :param V: 所有可能的观测状态的集合
        :param O: 观测序列
        :param A: 马尔科夫链的状态转移矩阵
        :param B: 观测状态生成的概率矩阵
        :param PI: 隐藏状态的初始概率分布
        :return: P(O|λ)
        """
        """
        Q = [1, 2, 3]
        V = ['红', '白']
        A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
        # O = ['红', '白', '红', '红', '白', '红', '白', '白']
        O = ['红', '白', '红', '白']   观察到的序列
        PI = [0.2, 0.4, 0.4]   
        """

        N = len(Q) # 隐藏状态序列长度
        M = len(O) # 观测状态序列长度
        alphas = np.zeros((N,M)) # 初始化前向概率
        #       [[0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0.]]
        # 行数为隐藏状态数，列数为时刻数
        # 要计算各个时刻的各个隐藏状态的前向概率
        T = M # 时刻数
        # 算法原理 https://s1.ax1x.com/2020/10/21/B9r63Q.png
        for t in range(T):
            index_O = V.index(O[t]) #当前观测状态在观测集合中的索引
            for i in range(N): # 计算各个隐藏状态的前向概率,
                if t==0:
                    alphas[i,t] = PI[i] * B[i,index_O]
                #            红    白
                # 隐藏状态1  [[0.5, 0.5],
                # 隐藏状态2  [0.4, 0.6],
                # 隐藏状态3  [0.7, 0.3]]
                else:
                    alphas[i,t] = np.dot([alpha[t - 1] for alpha in alphas],[a[i] for a in A]) * B[i,index_O]
                    # https://s1.ax1x.com/2020/10/21/BCuJiT.png

        self.forword_p = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas

    def backward(self, Q, V, A, B, O, PI):
        N = len(Q) # 隐藏状态序列长度
        M = len(O) # 观测状态序列长度
        betas = np.zeros((N,M)) # 初始化后向概率
        #算法原理 https://s1.ax1x.com/2020/10/21/BC1AC6.png

        for t in range(M - 2, -1, -1): # 对观测序列逆向遍历
            index_O = V.index(O[t+1])  # 观测为t+1的
            for i in range(N): # 通过t+1时刻N个状态的后向概率计算t时刻i状态的后向概率
                betas[i,t] = np.dot()




