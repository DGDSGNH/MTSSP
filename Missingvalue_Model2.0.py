import torch
import copy
import Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from torch.utils.data import Dataset,DataLoader,TensorDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


RANDOM_SEED = Config.RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
batch_size = Config.batch_size
input_size = Config.input_size
embed_dim = Config.embed_dim
num_layers = Config.num_layers
hidden_size = Config.hidden_size
workers = Config.workers
learning_rate = Config.learning_rate
epochs = Config.epochs
device = Config.device
datatype = Config.datatype
save_model_dir = '/home/lb/Model/mimic_Missingvaluemodel_'+datatype+'.pth'  # 模型保存的路径
testepoch = 10
if datatype=="mimic3":
    batch_size = 256
    learning_rate = 1e-2
else:
    batch_size = 512
    learning_rate = 1e-3
    testepoch = 5

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

def get_data():
    x_torch = pickle.load(open('dataset/lb_'+datatype+'_x_for_missingvalue.p', 'rb'))
    m_torch = pickle.load(open('dataset/lb_'+datatype+'_m_for_missingvalue.p', 'rb'))
    delta_torch = pickle.load(open('dataset/lb_'+datatype+'_delta_for_missingvalue.p', 'rb'))
    y_torch = pickle.load(open('dataset/lb_'+datatype+'_y.p', 'rb'))
    x_lens = pickle.load(open('dataset/lb_'+datatype+'_len.p', 'rb'))
    print(x_torch.shape)
    print(m_torch.shape)
    print(delta_torch.shape)
    print(y_torch.shape)
    print(x_lens.shape)

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    N = len(x_torch)

    training_x = x_torch[: int(train_ratio * N)]
    validing_x = x_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_x = x_torch[int((train_ratio + valid_ratio) * N):]

    training_m = m_torch[: int(train_ratio * N)]
    validing_m = m_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_m = m_torch[int((train_ratio + valid_ratio) * N):]

    training_delta = delta_torch[: int(train_ratio * N)]
    validing_delta = delta_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_delta = delta_torch[int((train_ratio + valid_ratio) * N):]

    training_x_lens = x_lens[: int(train_ratio * N)]
    validing_x_lens = x_lens[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_x_lens = x_lens[int((train_ratio + valid_ratio) * N):]

    training_y = y_torch[: int(train_ratio * N)]
    validing_y = y_torch[int(train_ratio * N): int(
        (train_ratio + valid_ratio) * N)]
    testing_y = y_torch[int((train_ratio + valid_ratio) * N):]

    train_deal_dataset = TensorDataset(training_x, training_y, training_m, training_delta, training_x_lens)

    train_loader = DataLoader(dataset=train_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    test_deal_dataset = TensorDataset(testing_x, testing_y, testing_m, testing_delta, testing_x_lens)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    valid_deal_dataset = TensorDataset(validing_x, validing_y, validing_m, validing_delta, validing_x_lens)

    valid_loader = DataLoader(dataset=valid_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    return train_loader, test_loader, valid_loader



class GRU_B(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU_B, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_xr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mr = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_xz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mz = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_xh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_dh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_mh = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

        self.linearfordelta = nn.Sequential(
            nn.Sigmoid()
        )
        self.linearforh = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, M, Delta):
        input = input.to(device).float()
        M = M.to(device).float()
        Delta = Delta.to(device).float()
        batchsize = input.size(0)
        step_size = input.size(1)

        listh_outs = []
        Delta = self.linearfordelta(Delta)
        Delta = 1 - Delta

        h1 = torch.zeros(batchsize, self.hidden_size).float().to(device)
        for i in range(step_size):
            delta = torch.squeeze(Delta[:, i:i + 1, :])  # batchsize,inputsize
            m = torch.squeeze(M[:, i:i + 1, :])
            x1 = torch.squeeze(input[:, i:i + 1, :])
            if i != 0:  # batchsize,inputsize
                x1 = m * x1 + (1 - m) * self.linearforh(h1) * delta

            z = torch.sigmoid((torch.mm(x1, self.w_xz) + torch.mm(h1, self.w_hz) + torch.mm(delta, self.w_dz) + torch.mm(m, self.w_mz) + self.b_z))  # batchsize,,hiddensize
            r = torch.sigmoid((torch.mm(x1, self.w_xr) + torch.mm(h1, self.w_hr) + torch.mm(delta, self.w_dr) + torch.mm(m, self.w_mr) + self.b_r))  # batchsize,,hiddensize
            h1_tilde = torch.tanh(
                (torch.mm(x1, self.w_xh) + torch.mm(r * h1, self.w_hh) + torch.mm(delta, self.w_dh) + torch.mm(m, self.w_mh) + self.b_h))  # batchsize,,hiddensize
            h1 = (1 - z) * h1 + z * h1_tilde  # batchsize,,hiddensize

            listh_outs.append(h1)

        h_outs = torch.stack(listh_outs).permute(1, 0, 2)  # batchsize,seqlen,inputsize
        #print(h_outs.shape)

        return h_outs



class MissingvalueModel(nn.Module):

    def __init__(self, input_size, hidden_size,output_size=1):
        super(MissingvalueModel, self).__init__()

        self.gru_b = GRU_B(input_size,hidden_size,output_size)
        self.gru_b2 = GRU_B(input_size,hidden_size,output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linearh2x = nn.Sequential(
            nn.Linear(2 * hidden_size, input_size)
        )

        self.linearfordelta = nn.Sequential(
            nn.Sigmoid()
        )

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        #padding=(kernel_size-1) * dilation_size
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=1, dilation=1)
        self.chomp1 = Chomp1d(1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=2, dilation=2)
        self.chomp2 = Chomp1d(2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=4, dilation=4)
        self.chomp3 = Chomp1d(4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(in_channels=embed_dim, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=8, dilation=8)
        self.chomp4 = Chomp1d(8)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)


        self.convseq1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.convseq2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.convseq3 = nn.Sequential(self.conv3, self.chomp3, self.relu3, self.dropout3)
        self.convseq4 = nn.Sequential(self.conv4, self.chomp4, self.relu4, self.dropout4)

        self.embedding = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.ReLU()
        )

        self.Wc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh()
        )
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)



    def forward(self, input,M,Delta,lens):
        input = input.to(device).float()
        input2 = torch.flip(input,[1])

        M = M.to(device)
        m2 = torch.flip(M,[1])

        Delta = Delta.to(device).float()
        delta2 = torch.flip(Delta,[1])

        lens = lens.to(device)

        h_outs_forward = self.gru_b(input,M,Delta) #batchsize,seqlen,hiddensize
        h_outs_reverse = self.gru_b2(input2,m2,delta2)
        h_outs_reverse2 = torch.flip(h_outs_reverse,[1])

        h_outs_bi = torch.cat((h_outs_forward,h_outs_reverse2),2)

        x_preds = self.linearh2x(h_outs_bi)  # inputsize

        v = self.embedding(x_preds)  # batchsize,seq_len,embdim
        pack = nn_utils.rnn.pack_padded_sequence(v, lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.rnn(pack)
        out_unpacked = nn_utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        outs, outlens = out_unpacked #batchsize,seqlen,2*hiddensize

        pack2 = nn_utils.rnn.pack_padded_sequence(v, lens.cpu(), batch_first=True, enforce_sorted=False)
        unpack2 = nn_utils.rnn.pad_packed_sequence(pack2, batch_first=True)
        v_conv = unpack2[0].permute(0, 2, 1) #batchsize,seqlen,2*hiddensize
        v_conv1 = self.convseq1(v_conv).permute(0, 2, 1)
        v_conv2 = self.convseq2(v_conv).permute(0, 2, 1)
        v_conv3 = self.convseq3(v_conv).permute(0, 2, 1)
        v_conv4 = self.convseq4(v_conv).permute(0, 2, 1)
        totalvconv = v_conv1 + v_conv2 + v_conv3 + v_conv4

        out3 = torch.mul(totalvconv, outs)
        
        out3 = self.Wc(out3)

        out3 = self.predict(out3)

        return out3,x_preds


class My_mse_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, x, x_preds):
        return torch.mean(torch.pow((M * x - M * x_preds), 2))


model = MissingvalueModel(input_size,hidden_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model,train_loader,valid_loader):
    model.train()
    train_loss_array = []
    Early_stopping = Config.EarlyStopping()

    for epoch in range(epochs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            out,xpreds = model(inputs,m,delta,lens)
            out = out.to(device)
            batch_loss = torch.tensor(0, dtype=float).to(device)
            for j in range(len(lens)):
                intlenj = int(lens[j])

                oneloss = torch.tensor(0).to(device)
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                # 下面是计算的一个样本的loss
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                batch_loss += oneloss

            batch_loss /= batch_size
            lossF2 = My_mse_loss()
            loss2 = lossF2(m,inputs,xpreds).to(device)

            losstotal = batch_loss+loss2

            optimizer.zero_grad()
            losstotal.backward(retain_graph=True)
            optimizer.step()
        if epoch % 4 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        if (epoch + 1) % 1 == 0:  # 每 1 次输出结果
            print('Epoch: {}, Train Loss: {}'.format(epoch + 1, losstotal.detach().data))
            train_loss_array.append(losstotal.detach().data)
            device = torch.device("cpu")
            model.eval()
            valid_losses = []
            for i, data in enumerate(valid_loader):
                inputs, labels, m, delta, lens = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                m = m.to(device)
                delta = delta.to(device)
                lens = lens.to(device)
                labels = labels.float()

                out, xpreds = model(inputs, m, delta, lens)
                out = out.to(device)
                xpreds = xpreds.to(device)
                batch_loss = torch.tensor(0, dtype=float).to(device)
                for j in range(len(lens)):  # 遍历256个样本
                    intlenj = int(lens[j])  # 这256个样本里的第j个样本的原本的长度
                    lossF = torch.nn.BCELoss(size_average=True).to(device)
                    oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0)).to(device)
                    batch_loss += oneloss

                batch_loss /= batch_size

                lossF2 = My_mse_loss()
                loss2 = lossF2(m, inputs, xpreds)

                losstotal = batch_loss + loss2
                valid_losses.append(losstotal.detach().data)


            valid_loss = np.average(valid_losses)
            print('Epoch: {}, Valid Loss: {}'.format(epoch + 1, valid_loss))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            Early_stopping(valid_loss, model, state, save_model_dir)

            if Early_stopping.early_stop:
                print("Early stopping")
                break

class My_rmse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, x, x_preds):
        return torch.sqrt(torch.mean(torch.pow((M * x - M * x_preds), 2)))

def test_model(model,test_loader):
    device = torch.device("cpu")
    model.eval()
    test_loss_array = []
    test_loss_array2 = []
    outs = list()
    labelss = list()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, m, delta, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            out,xpreds = model(inputs,m,delta, lens)

            out = out.to(device)

            batch_loss = torch.tensor(0, dtype=float).to(device)
            for j in range(len(lens)):
                intlenj = int(lens[j])
                oneloss = torch.tensor(0).to(device)
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                outs.extend(list(out[j, intlenj - 1, :].numpy()))
                templabel = [int(labels[j])]
                labelss.extend(templabel)
                batch_loss += oneloss

            batch_loss /= batch_size

            lossF = torch.nn.BCELoss(size_average=True).to(device)
            xpreds = xpreds.to(device)
            lossF2 = My_mse_loss().to(device)
            loss2 = lossF2(m, inputs, xpreds)

            print('Test loss1:{}'.format(float(batch_loss.data)))
            test_loss_array.append(float(batch_loss.data))
            print('Test loss2:{}'.format(float(loss2.data)))
            test_loss_array2.append(float(loss2.data))




    outs = np.array(outs)
    labelss = np.array(labelss)

    auroc = metrics.roc_auc_score(labelss, outs)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labelss, outs)
    auprc = metrics.auc(recalls, precisions)

    return auroc, auprc


def mask10perdata(model):
    x_torch = pickle.load(open('dataset/lb_' + datatype + '_x_for_missingvalue.p', 'rb'))
    m_torch = pickle.load(open('dataset/lb_' + datatype + '_m_for_missingvalue.p', 'rb'))

    delta_torch = pickle.load(open('dataset/lb_' + datatype + '_delta_for_missingvalue.p', 'rb'))
    y_torch = pickle.load(open('dataset/lb_' + datatype + '_y.p', 'rb'))
    x_lens = pickle.load(open('dataset/lb_' + datatype + '_len.p', 'rb'))

    # 划分训练集验证集和测试集，8：1：1
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    N = len(x_torch)
    testing_x = x_torch[int((train_ratio + valid_ratio) * N):]
    testing_m = m_torch[int((train_ratio + valid_ratio) * N):]
    testing_delta = delta_torch[int((train_ratio + valid_ratio) * N):]
    testing_x_lens = x_lens[int((train_ratio + valid_ratio) * N):]
    testing_y = y_torch[int((train_ratio + valid_ratio) * N):]

    N = len(testing_m) #4629
    print(N)

    m = torch.reshape(testing_m, (-1,N * 63 * 35)).squeeze()
    timestepidx = []#把m=1的点都挑出来的下标列表
    timestepidx = pickle.load(open('lb_' + datatype + '_maskidx.p', 'rb'))

    print(len(timestepidx))#1133452个点是有值的

    timestep = torch.zeros_like(m) # 把m=1的点都挑出来的矩阵,随机遮蔽的点的位置最后置1 4629,2205


    #timestep = torch.reshape(timestep, (-1, N * 63 * 35)).squeeze()#因为np.random.choice只能处理一维数组
    print(timestep.shape)
    #timestep.numpy()  # list转numpy

    masktimestepidx = np.random.choice(timestepidx,len(timestepidx) // 10, replace=False).tolist()  # 从m=1的下标列表这里面随机选出10%的要遮蔽的点的下标
    print(len(masktimestepidx))
    for i in masktimestepidx:
        timestep[i] = 1#遮蔽这些点，遮蔽的点置1


    m = m-timestep #1-1=0，这就实现了把m矩阵对应的点变为0的工作
    #把m和timestep还原成三维，
    m = torch.reshape(m,(N,35,63))
    timestep = torch.reshape(timestep,(N,35,63))
    testing_x_mask = (1-timestep)*testing_x #同时也实现了把输入对应的点变为0的工作
    #testing_x是真实值输入矩阵，testing_x_mask是遮蔽之后对应的值变为0的矩阵
    print(timestep.shape)
    print(testing_x.shape)
    print(m.shape)

    test_deal_dataset = TensorDataset(testing_x,testing_x_mask, testing_y, m, testing_delta, testing_x_lens,timestep)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    device = torch.device("cpu")
    model.eval()
    k=0
    totalrmse = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs,inputsmask, labels, m, delta, lens,timestepbatch = data
            inputs = inputs.to(device)
            inputsmask = inputsmask.to(device)
            labels = labels.to(device)
            m = m.to(device)
            delta = delta.to(device)
            lens = lens.to(device)
            labels = labels.float()

            # print(inputs.shape)
            # 前向传播
            out, xpreds = model(inputsmask, m, delta, lens)

            xpreds = xpreds.to(device)

            rmsefunction = My_rmse().to(device)
            rmse = rmsefunction(timestepbatch, inputs, xpreds)
            totalrmse = totalrmse+rmse.item()
            k = k+1

    finalrmse = float(totalrmse/k)
    print(finalrmse)
    return finalrmse


def main():
    train_loader,test_loader,valid_loader = get_data()
    train_model(model, train_loader, valid_loader)
    checkpoint = torch.load(save_model_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epoch']

    aurocs = []
    auprcs = []
    for i in range(testepoch):
        train_loader, test_loader, valid_loader = get_data()
        auroc,auprc,rmse = test_model(model, test_loader)
        aurocs.append(auroc)
        auprcs.append(auprc)

    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs,ddof=1)
    auprc_mean = np.mean(auprcs)
    auprc_std = np.std(auprcs, ddof=1)
    print("auroc 平均值为：" + str(auroc_mean) + " 标准差为：" + str(auroc_std))
    print("auprc 平均值为：" + str(auprc_mean) + " 标准差为：" + str(auprc_std))

    mask10perdata(model)


    return

if __name__ == '__main__':
    main()

