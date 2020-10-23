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
        # 代码逻辑 https://s1.ax1x.com/2020/10/21/BCuJiT.png
        for t in range(T):
            index_O = V.index(O[t]) #当前观测状态在观测集合中的索引
            print(f"time {t + 1} and observe state is {O[t]}")  # 因为这边python的索引是从0开始的
            for i in range(N): # 计算各个隐藏状态的前向概率,
                if t==0:
                    alphas[i,t] = PI[i] * B[i,index_O]
                #            红    白
                # 隐藏状态1  [[0.5, 0.5],
                # 隐藏状态2  [0.4, 0.6],
                # 隐藏状态3  [0.7, 0.3]]
                else:
                    alphas[i,t] = np.dot([alpha[t - 1] for alpha in alphas],[a[i] for a in A]) * B[i,index_O]
                if t == T-1:
                        print(f"time {t+1} state {Q[i]} alpha is {alphas[i, t]:.4f}")

        self.forword_p = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas
        print(f"finally p is {self.forword_p:.4f}")

    def backward(self, Q, V, A, B, O, PI):
        N = len(Q) # 隐藏状态序列长度
        M = len(O) # 观测状态序列长度
        betas = np.ones((N,M)) # 初始化后向概率
        # 算法原理 https://s1.ax1x.com/2020/10/21/BC1AC6.png
        # 代码逻辑 https://s1.ax1x.com/2020/10/21/BPkLGj.jpg
        for t in range(M - 2, -1, -1): # 对观测序列逆向遍历
            index_O = V.index(O[t+1])  # 观测为t+1的
            print(f"time {t+1} and observe state is {O[t+1]}") # 因为这边python的索引是从0开始的
            for i in range(N): # 通过t+1时刻N个状态的后向概率计算t时刻i状态的后向概率
                betas[i,t] = np.dot(np.multiply(A[i],[b[index_O] for b in B]),[beta[t+1] for beta in betas])
                if t == 0:
                    print(f"time {t+1} state {Q[i]} beta is {betas[i,t]:.4f}")

        index_first = V.index(O[0]) # 第一个观测在观测集合中的索引
        self.beats = betas
        p = np.dot(np.multiply(PI,[b[index_first] for b in B]),[beta[0] for beta in betas])
        self.backward_p = p
        print(f"pi is {PI}")
        print(f"finally p is {self.backward_p:.4f}")

    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q) # 隐藏状态序列长度
        M = len(O) # 观测状态序列长度
        deltas = np.zeros((N, M)) # 局部状态1
        I = np.zeros(M) # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        psis = np.zeros((N, M)) # 初始化psis
        # 算法原理 https://s1.ax1x.com/2020/10/21/BPut3D.png
        # 计算例子 https://s1.ax1x.com/2020/10/21/BPuWuj.jpg
        # 代码逻辑
        for t in range(M):
            index_O = V.index(O[t])
            print(f"time {t + 1} and observe state is {O[t]}")  # 因为这边python的索引是从0开始的
            for i in range(N):
                if t == 0: # 初始化局部状态
                    deltas[i,t] = PI[i] * B[i,index_O]
                    psis[i,t] = 0
                else:
                    deltas[i, t] = np.max(np.multiply([delta[t - 1] for delta in deltas],
                                                      [a[i] for a in A])) * B[i,index_O]
                    psis[i,t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas],
                                                      [a[i] for a in A]))
                print(f"hidden state {i+1} max deltas is {deltas[i, t]:.4f} and psis is {psis[i, t]}")

        I[M - 1] = np.argmax([delta[M - 1] for delta in deltas]) # 最后时刻，概率最大的隐状态
        for t in range(M - 2, -1, -1): # 递归由后向前得到其他结点
            I[t] = psis[int(I[t+1]),t+1]

        print(f"finally sequence is {I+1}") # python的索引是从0开始的







