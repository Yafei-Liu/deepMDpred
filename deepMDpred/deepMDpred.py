# coding=utf-8
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import timeit
import csv
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import uniform_

from sklearn.metrics import roc_auc_score

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class mdprediction(nn.Module):
    def __init__(self):
        super(mdprediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint + 1, fm)
        self.W_sub = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(radius)])
        self.fc = nn.Linear(n_seqFeature, dim)
        uniform_(self.fc.weight, -0.01, 0.01)
        self.fc1 = nn.Linear(333, dim)
        self.fc2 = nn.Linear(20, dim)

        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, n_classes)
        self.gcn_x1 = GCNConv(fm, fm)
        self.gcn_x2 = GCNConv(fm, dim)

    def forward(self, inputs):

        fingerprints, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency, words, sim_vector = inputs
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        adjacency1 = adjacency.nonzero()
        adjacency1 = adjacency1.T
        x_m_d1 = torch.relu(self.gcn_x1(fingerprint_vectors, adjacency1, adjacency[adjacency1[0], adjacency1[1]]))
        x_m_d3 = torch.relu(self.gcn_x2(x_m_d1, adjacency1, adjacency[adjacency1[0], adjacency1[1]]))
        attention1 = nn.Linear(x_m_d3.size(1), x_m_d3.size(1))
        x_m_d2 = attention1(x_m_d3)
        x_m_d2 = torch.relu(x_m_d2)
        attention2 = nn.Linear(x_m_d3.size(1), 1)
        x_m_d2 = attention2(x_m_d2)
        x_m_d2 = torch.sigmoid(x_m_d2)
        disease_vector = x_m_d2 * x_m_d3
        disease_vector = torch.unsqueeze(torch.mean(disease_vector, 0), 0)
        sim_vector = sim_vector.view(1, -1)
        disease_sim = self.fc1(sim_vector)
        cat_disease = torch.cat((disease_vector, disease_sim), 1)
        disease_vector = self.fc2(cat_disease)

        """miRNA vector with MLP."""
        words = words.view(1, -1)
        mirna_vector = self.fc(words)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((disease_vector, mirna_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        association = self.W_interaction(cat_vector)
        return association

    def __call__(self, data, train=True):

        inputs, correct_association = data[:-1], data[-1]
        predicted_association = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_association, correct_association)
            return loss
        else:
            correct_labels = correct_association.to('cpu').data.numpy()

            ys = F.softmax(predicted_association, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = ys
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, HTMDGCN):
        self.model = HTMDGCN
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, data, mirnafeature, matrix, diseasefeature):
        np.random.shuffle(data)
        loss_total = 0
        for data_i, datain in enumerate(data):
            one_mirnaid, one_diseaseid, interaction = datain[1], datain[0], datain[-1]
            mirna_feature = mirnafeature[int(one_mirnaid), :]
            mirna_feature = np.array(list(mirna_feature), dtype=np.float32)
            dds_sim = matrix[int(one_diseaseid), :]
            dds_sim = np.array(list(dds_sim))
            finger, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_new = diseasefeature[
                int(one_diseaseid)]
            atom_degree_list = torch.from_numpy(atom_degree_list).to(device)
            bond_feature = torch.from_numpy(bond_feature).to(device)
            bond_degree_list = torch.from_numpy(bond_degree_list).to(device)
            i_bond_j = torch.from_numpy(i_bond_j).to(device)
            finger_tensor = torch.from_numpy(finger).to(device)
            adjacency_tensor = torch.from_numpy(adjacency_new).to(device).float()
            mirna_feature_tensor = torch.from_numpy(mirna_feature).to(device)
            dds_sim_tensor = torch.from_numpy(dds_sim).to(device)
            datafeaturein = (
                finger_tensor, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_tensor,
                mirna_feature_tensor, dds_sim_tensor, interaction)
            loss = self.model(datafeaturein)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, data, mirnafeature, matrix, diseasefeature):
        t, y, s = [], [], []
        for datain in data:
            one_mirnaid, one_diseaseid, interaction = datain[1], datain[0], datain[-1]
            mirna_feature = mirnafeature[int(one_mirnaid), :]
            mirna_feature = np.array(list(mirna_feature))
            dds_sim = matrix[int(one_diseaseid), :]
            dds_sim = np.array(list(dds_sim))
            finger, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_new = diseasefeature[
                int(one_diseaseid)]
            atom_degree_list = torch.from_numpy(atom_degree_list).to(device)
            bond_feature = torch.from_numpy(bond_feature).to(device)
            bond_degree_list = torch.from_numpy(bond_degree_list).to(device)
            i_bond_j = torch.from_numpy(i_bond_j).to(device)
            finger_tensor = torch.from_numpy(finger).to(device)
            adjacency_tensor = torch.from_numpy(adjacency_new).to(device).float()
            mirna_feature_tensor = torch.from_numpy(mirna_feature).to(device)
            dds_sim_tensor = torch.from_numpy(dds_sim).to(device)
            datafeaturein = (
                finger_tensor, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_tensor,
                mirna_feature_tensor, dds_sim_tensor, interaction)
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(datafeaturein, train=False)
            t.append(correct_labels)
            y.append(predicted_labels)
            s.append(predicted_scores)
        y_one_hot = label_binarize(y=t, classes=np.arange(n_classes))
        #y_one_hot = to_categorical(y_one_hot, 2)
        s = np.array(s)
        s = np.reshape(s, (-1, n_classes))
        auc1 = roc_auc_score(y_one_hot, s, average='macro')
        auc_per = per(y_one_hot, s)
        return auc1, auc_per

    def save_result(self, result, filename):
        with open(filename, 'a') as ff:
            ff.write('\t'.join(map(str, result)) + '\n')


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def load_pickle(file_name):
    with open(file_name, 'rb') as ff:
        return pickle.load(ff)


def load_strlist(file_name):
    readlist = []
    with open(file_name, 'r') as ff:
        for line in ff:
            readlist.append(line.strip())
    ff.close()
    return readlist


def load_matrix(file_name):
    with open(file_name, "r") as inf:
        matrix = [line.strip("\n").split()[0:] for line in inf]
    inf.close()
    matrix = np.array(matrix, dtype=np.float32)
    return matrix


def shuffle_dataset(data, seed):
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


def split_dataset(data, ratio):
    n = int(ratio * len(data))
    dataset_1, dataset_2 = data[:n], data[n:]
    return dataset_1, dataset_2


def per(y_one_hot, y_pred):
    auc1 = np.zeros(n_classes)
    for i in range(n_classes):
        auc1[i] = roc_auc_score(y_one_hot[:, i], y_pred[:, i])
    return auc1


if __name__ == "__main__":
    DATASET = 'miRNAdisease'
    radius = 2
    ngram = 3
    dim = 10
    layer_gnn = 3
    side = 5
    window = (2 * side + 1)
    layer_cnn = 3
    layer_output = 3
    lr = 1e-3
    lr_decay = 0.5
    decay_interval = 30
    weight_decay = 1e-6
    iteration = 100
    # fm=128
    fm = 10
    # fm=256
    # fm=20
    # fm=5
    AUC = []
    #n_classes = 4
    # n_classes = 5
    n_classes = 6
    # n_classes = 2

    AUC_per_MLP = np.zeros(n_classes)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data1."""
    # dir_input = './miRNAdisease-task1/'
    # CV_mirnaid_list = load_strlist(dir_input + 'miRNA_type1.txt')
    # mirnafeaturemat = load_matrix(dir_input + 'miRNA_fea.txt')
    # n_seqFeature = len(mirnafeaturemat[0, :])
    # CV_diseaseid_list = load_strlist(dir_input + 'disease_type1.txt')
    # diseasefeaturemat = np.load(dir_input + "/disease_fea/disease_fea.npy", allow_pickle=True)
    # dd_matrix1 = read_csv(dir_input + '/disease_sem.csv')
    # dd_matrix2 = read_csv(dir_input + '/disease_func.csv')
    # dd_s_matrix = dd_matrix1 + dd_matrix2
    # interactions = load_tensor(dir_input + 'label_types', torch.LongTensor)
    # fingerprint_dict = load_pickle(dir_input + 'fingerprint.pickle')
    # n_fingerprint = len(fingerprint_dict)
    # print(len(interactions))

    """Load preprocessed data2."""
    # dir_input = ('./miRNAdisease-task2/'
    # CV_mirnaid_list = load_strlist(dir_input + 'miRNA_type2.txt')
    # CV_diseaseid_list = load_strlist(dir_input + 'disease_type2.txt')
    # associations = load_tensor(dir_input + 'label_types', torch.LongTensor)
    # mirnafeaturemat = load_matrix(dir_input + 'miRNA_fea.txt')
    # n_seqFeature = len(mirnafeaturemat[0, :])
    # diseasefeaturemat = np.load(dir_input + "disease_fea.npy", allow_pickle=True)
    # dd_matrix1 = read_csv(dir_input + '/disease_sem.csv')
    # dd_matrix2 = read_csv(dir_input + '/disease_func.csv')
    # dd_s_matrix = dd_matrix1 + dd_matrix2
    # fingerprint_dict = load_pickle(dir_input + 'fingerprint.pickle')
    # n_fingerprint = len(fingerprint_dict)
    # print(len(associations))

    """Load preprocessed data3."""
    dir_input = ('./miRNAdisease-task3/')
    CV_mirnaid_list = load_strlist(dir_input + 'miRNA_type3.txt')
    CV_diseaseid_list = load_strlist(dir_input + 'disease_type3.txt')
    associations = load_tensor(dir_input + 'label_types', torch.LongTensor)
    mirnafeaturemat = load_matrix(dir_input + 'miRNA_fea.txt')
    n_seqFeature = len(mirnafeaturemat[0, :])
    dd_matrix1 = read_csv(dir_input + '/disease_sem.csv')
    dd_matrix2 = read_csv(dir_input + '/disease_func.csv')
    dd_s_matrix = dd_matrix1 + dd_matrix2
    diseasefeaturemat = np.load(dir_input + "disease_fea.npy", allow_pickle=True)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint.pickle')
    n_fingerprint = len(fingerprint_dict)
    print(len(associations))

    """Load preprocessed data4."""
    # dir_input = ('./miRNAdisease-task4/')
    # CV_mirnaid_list = load_strlist(dir_input + 'miRNA_type4.txt')
    # CV_diseaseid_list = load_strlist(dir_input + 'disease_type4.txt')
    # associations = load_tensor(dir_input + 'label_types.npy', allow_pickle=True)
    # #print(associations)
    # mirnafeaturemat = load_matrix(dir_input + 'miRNA_fea.txt')
    # n_seqFeature = len(mirnafeaturemat[0, :])
    # diseasefeaturemat = np.load(dir_input + "disease_fea.npy", allow_pickle=True)
    # fingerprint_dict = load_pickle(dir_input + 'fingerprint.pickle')
    # n_fingerprint = len(fingerprint_dict)
    # dd_matrix1 = read_csv(dir_input + '/disease_sem.csv')
    # dd_matrix2 = read_csv(dir_input + '/disease_func.csv')
    # dd_s_matrix = dd_matrix1 + dd_matrix2
    # print(len(associations))

    """Create a dataset and split it into train/dev/test."""
    # dataset = list(zip(CV_diseaseid_list, CV_mirnaid_list, associations))

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(CV_diseaseid_list, CV_mirnaid_list, interactions))
    fold_time = 0
    dataset = shuffle_dataset(dataset, 23)
    for kk in range(1):
        """Output files."""
        file_output = './output/task-result.txt'
        Results = 'Epoch\tTime(sec)\tLoss_train'
        with open(file_output, 'w') as f:
            f.write(Results + '\n')
        dataset = np.array(dataset)
        skf = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)
        for train_index, test_index in skf.split(dataset, dataset[:, -1]):
            fold_time = fold_time + 1
            print(fold_time)
            dataset_train, dataset_test = dataset[train_index], dataset[test_index]
            """Set a model."""
            torch.manual_seed(1234)
            HTMDGCN = mdprediction().to(device)
            trainer = Trainer(HTMDGCN)
            tester = Tester(HTMDGCN)

            """Start training."""
            print('Training...')
            print(Results)
            start = timeit.default_timer()
            for epoch in range(1, 100):
                print(epoch)
                if epoch % decay_interval == 0:
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay
                loss_train = trainer.train(dataset_train, mirnafeaturemat, dd_s_matrix,
                                           diseasefeaturemat)
                AUC_test, AUC_per = tester.test(dataset_test, mirnafeaturemat, dd_s_matrix, diseasefeaturemat)
                end = timeit.default_timer()
                time = end - start
                Results = [epoch, time, loss_train]
                tester.save_result(Results, file_output)
                print('\t'.join(map(str, Results)))




