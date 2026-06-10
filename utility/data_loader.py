"""
Created on April 10, 2021
PyTorch Implementation of GNN-based Recommender System
This file is used to read users, items, interaction information
"""
import numpy as np
import os
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


class Data(object):
    def __init__(self, path, config):
        self.path = path
        self.num_users = 0
        self.num_items = 0
        self.num_entities = 0
        self.num_relations = 0
        self.num_nodes = 0
        self.num_train = 0
        self.num_test = 0

        self.load_data()
        if config:
            self.split_test_dict = None
            self.split_state = None
            if int(config.sparsity_test) == 1:
                self.split_test_dict, self.split_state = self.create_sparsity_split()

    def load_data(self):
        train_path = self.path + "/train.txt"
        test_path = self.path + "/test.txt"

        train_user, self.train_user, self.train_item, self.num_train, self.pos_length = self.read_ratings(train_path)

        test_user, self.test_user, self.test_item, self.num_test, _ = self.read_ratings(test_path)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        self.data_statistics()

        assert len(self.train_user) == len(self.train_item)

        # 构建用户-物品交互矩阵（稀疏矩阵表示）
        # 行表示用户，列表示物品
        # 如果用户 u 与物品 i 有交互，则 (u, i) 位置为 1
        self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                           shape=(self.num_users, self.num_items))

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))  # 获取每个用户对应的正样本物品列表

        # 构建测试集字典（通常用于评测阶段）
        # key: 用户 ID
        # value: 该用户在测试集中的正样本物品
        self.test_dict = self.build_test()

        #add noise

#         self.train_user, self.train_item = self.add_noise(0.1)
#         self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
#                                            shape=(self.num_users, self.num_items))
#         self.num_train = len(self.train_user)
#         self.data_statistics()

    def read_ratings(self, file_name):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()  # 去除首尾空白字符
                arr = [int(i) for i in temp.split(" ")]  # 按空格切分，并转为 int

                # arr[0] 是用户 ID，arr[1:] 是该用户对应的正样本物品 ID 列表
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)

                # 如果该用户没有正样本物品，直接跳过
                if len(pos_id) < 1:
                    line = f.readline()
                    continue
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
                inter_users.extend([user_id] * len(pos_id))  # 将当前用户 ID 按其正样本数量重复，加入交互用户列表
                pos_length.append(len(pos_id))  # 记录该用户的正样本物品数量
                inter_items.extend(pos_id)  # 将正样本物品加入交互物品列表
                inter_num += len(pos_id)  # 更新交互总数
                line = f.readline()

        # 返回：
        # unique_users : 唯一用户 ID 列表
        # inter_users  : 按交互展开的用户 ID 列表
        # inter_items  : 按交互展开的物品 ID 列表
        # inter_num    : 总交互数
        # pos_length   : 每个用户的正样本物品数量
        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

    def data_statistics(self):
        print("\t num_users:", self.num_users)
        print("\t num_items:", self.num_items)
        print("\t num_nodes:", self.num_nodes)
        print("\t num_train:", self.num_train)
        print("\t num_test: ", self.num_test)

        # 计算并打印数据集的稀疏度（Sparsity）
        # 稀疏度 = 1 - (已有交互数 / 用户数 / 物品数)
        # 反映用户-物品交互矩阵的稀疏程度
        print("\t sparisty: ", 1 - (self.num_train + self.num_test) / self.num_users / self.num_items)

    # random sampling from official implementation of LightGCN
    def sample_data_to_train_random(self):
        users = np.random.randint(0, self.num_users, len(self.train_user))
        sample_list = []
        for i, user in enumerate(users):
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_index = np.random.randint(0, len(positive_items))
            positive_item = positive_items[positive_index]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)

    def sample_data_to_train_all(self):
        sample_list = []

        # 遍历训练集中所有 (user, item) 正样本交互
        for i in range(len(self.train_user)):
            user = self.train_user[i]

            # 获取该用户的所有正样本物品集合（训练集中出现过的物品）
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue

            positive_item = self.train_item[i]  # 当前索引 i 对应的正样本物品

            # ----------------- 负采样过程 -----------------
            # 从所有物品中随机采样一个负样本
            # 要求该物品不在用户的正样本集合中
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    # 如果采样到的是正样本，则重新采样
                    continue
                else:  # 采样到合法负样本，跳出循环
                    break

            # 将 (user, positive_item, negative_item) 组成一个训练样本
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)

    def get_user_pos_items(self, users):
        positive_items = []
        for user in users:
            # 从用户-物品交互稀疏矩阵中取出该用户对应的一行
            # nonzero()[1] 返回该用户有过交互的物品索引（列索引）
            positive_items.append(self.user_item_net[user].nonzero()[1])
        return positive_items

    def get_user_n_neg_items(self, users, n):
        negative_items = []
        for user in users:
            negative_list = []
            for i in range(n):
                while True:
                    negative_item = np.random.randint(0, self.num_items)
                    if negative_item in self.all_positive[user]:
                        continue
                    else:
                        negative_list.append(negative_item)
                        break
            negative_items.append(negative_list)

        return negative_items

    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def sparse_adjacency_matrix_with_self(self):
        try:
            norm_adjacency = sp.load_npz(self.path + '/pre_A_with_self.npz')
            print("\t Adjacency matrix loading completed.")
        except:
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.user_item_net.todok()

            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T
            adjacency_matrix = adjacency_matrix.todok()
            adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_A_with_self', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        return norm_adjacency

    def sparse_adjacency_matrix(self):
        try:
            # 尝试从磁盘加载已经预处理并保存好的归一化邻接矩阵
            norm_adjacency = sp.load_npz(self.path + '/pre_A.npz')
            # print("\t Adjacency matrix loading completed.")
        except:
            # 如果不存在预处理文件，则从头构建邻接矩阵

            # 初始化一个 (num_nodes × num_nodes) 的稀疏邻接矩阵
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()

            # 将用户-物品交互矩阵转为 DOK 格式
            R = self.user_item_net.todok()

            # 构建二部图的邻接关系
            # 左上角：用户-用户（为空）
            # 右下角：物品-物品（为空）
            # 右上角：用户 → 物品
            adjacency_matrix[:self.num_users, self.num_users:] = R
            # 坐下角：物品 → 用户
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            # 转换回 DOK 格式，便于后续操作
            adjacency_matrix = adjacency_matrix.todok()

            # 计算每个节点的度（行求和）
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            # 计算度矩阵 D^{-1/2}
            d_inv = np.power(row_sum, -0.5).flatten()
            # 将无穷大的值（度为 0 的节点）置为 0
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            # 对邻接矩阵进行对称归一化：D^{-1/2} A D^{-1/2}
            norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()

            # 将归一化后的邻接矩阵保存到磁盘，便于下次直接加载
            sp.save_npz(self.path + '/pre_A', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        # 返回归一化后的稀疏邻接矩阵
        return norm_adjacency
       
    def sparse_adjacency_matrix_adjnorm(self):
        try:
            norm_adjacency = sp.load_npz(self.path + '/pre_A_adjnorm.npz')
            print("\t Adjacency matrix loading completed.")
        except:
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.user_item_net.todok()

            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T
            adjacency_matrix = adjacency_matrix.todok()

            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.25).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)
            
            col_sum = np.array(adjacency_matrix.sum(axis=0))
            
            col_d_inv = np.power(row_sum, -0.75).flatten()
            col_d_inv[np.isinf(col_d_inv)] = 0.
            col_degree_matrix = sp.diags(col_d_inv)

            norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_A_adjnorm', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        return norm_adjacency
      
    def sparse_adjacency_matrix_R(self):
        try:
            norm_adjacency = sp.load_npz(self.path + '/pre_R.npz')
            print("\t Adjacency matrix loading completed.")
        except:
            adjacency_matrix = self.user_item_net

            row_sum = np.array(adjacency_matrix.sum(axis=1))
            row_d_inv = np.power(row_sum, -0.5).flatten()
            row_d_inv[np.isinf(row_d_inv)] = 0.
            row_degree_matrix = sp.diags(row_d_inv)

            col_sum = np.array(adjacency_matrix.sum(axis=0))
            col_d_inv = np.power(col_sum, -0.5).flatten()
            col_d_inv[np.isinf(col_d_inv)] = 0.
            col_degree_matrix = sp.diags(col_d_inv)

            norm_adjacency = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_R', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        return norm_adjacency
        
    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list() 
        temp = []
        count = 1
        fold = 4
        n_count = self.num_train + self.num_test
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.num_train + self.num_test):
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
       
    def add_noise(self, ratio):
        count = 0
        train_user = self.train_user.tolist()
        train_item = self.train_item.tolist()
        while count < self.num_train * ratio:
            user_id = np.random.randint(self.num_users)
            item_id = np.random.randint(self.num_items)

            if item_id not in self.all_positive[user_id]:
                if item_id not in self.test_dict[user_id]:
                    train_user.append(user_id)
                    train_item.append(item_id)
                    count += 1
        print(len(self.train_user.tolist()))
        print(count, "noise data have been added.")
        return np.array(train_user), np.array(train_item)



