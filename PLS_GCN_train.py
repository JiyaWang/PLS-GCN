import pickle
import pandas as pd
import torch
import numpy
from PLS_GCN_model import *
from PLS_GCN_model import _LossFunction
import argparse
import warnings
from tqdm import tqdm
from data_process import train_test
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F


# 求协方差矩阵的上三角
def process_covariance(x):
    co_numpy = x.detach().numpy()
    triangle = numpy.triu(co_numpy, k=1)
    y = torch.tensor(triangle)
    return y


class PLS_model:
    def __init__(self, data, device, optimizer, model, K, program, para):
        self.data = data.to(device)
        self.device = device
        self.optimizer = optimizer
        self.model = model
        self.K = K
        self.para = para
        self.train_process = './result/train/process/{}_program_{}_lr_{}_weight_{}'.format(
            datetime.datetime.now().strftime("%m_%d_%H"), program, para['lr'],
            para['weight'])
        self.model_file_path = './result/train/final/{}_method_{}_lr_{}_weight_{}.pth'.format(
            datetime.datetime.now().strftime("%m_%d_%H"), program, para['lr'], para['weight'])
        self.loss_file_path = "./result/loss/{}.method({})_train_loss+test_loss_lr_{}_weight_{}.csv".format(
            datetime.datetime.now().strftime("%m_%d_%H"), program, para['lr'], para['weight'])
        self.train_data, self.test_data = train_test(data, data.train_mask).to(device), train_test(data,
                                                                                                   data.test_mask).to(
            device)
        self.class_num = torch.max(data.y) + 1
        self.program = program

    def train(self):
        writer = SummaryWriter(log_dir='./runs/experiment')
        os.makedirs(self.train_process, exist_ok=True)
        # 选择模型
        if self.model == 'gcn':
            print('gcn')
            model = GCN(self.class_num, self.program)
        model.to(self.device)
        print(self.program)

        # 选择优化器
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.para['lr'], weight_decay=1e-4)
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.para['lr'], momentum=0.9, nesterov=True,
                                        weight_decay=1e-3)

        loss_fn = _LossFunction(torch.tensor(self.para['weight']).to(self.device))

        # 创建loss保存文件
        df = pd.DataFrame(
            columns=['train_losses', 'train_ce_losses', 'train_frobenius_losses', 'test_losses', 'test_ce_losses',
                     'test_frobenius_losses'])
        df.to_csv(self.loss_file_path, index=False)

        for epoch in tqdm(range(self.para['epoch'])):
            # 训练集
            model.train()
            optimizer.zero_grad()
            _, out, residual = model(self.train_data, self.K,
                                     self.device)  # Perform a single forward pass.
            loss, ce_loss, frobenius_loss = loss_fn(out, self.train_data.y,
                                                    self.train_data.x,
                                                    residual)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()

            train_losses = "%f" % loss
            train_ce_losses = "%f" % ce_loss
            train_frobenius_losses = "%f" % frobenius_loss

            # 测试集
            model.eval()
            _, out, residual = model(self.test_data, self.K, self.device)
            test_loss, test_ce_loss, test_frobenius_loss = loss_fn(out, self.test_data.y, self.test_data.x, residual)
            writer.add_scalars(
                'loss_lr:{}_weight:{}_epoch:{}_program_{}_again'.format(self.para['lr'], self.para['weight'],
                                                                        self.para['epoch'], self.program),
                {'train_loss': loss, 'test_loss': test_loss},
                global_step=epoch)

            test_losses = "%f" % test_loss
            test_ce_losses = "%f" % test_ce_loss
            test_frobenius_losses = "%f" % test_frobenius_loss

            # 保存数据（loss）
            list_loss = [train_losses, train_ce_losses, train_frobenius_losses, test_losses, test_ce_losses,
                         test_frobenius_losses]
            loss_csv = pd.DataFrame([list_loss])
            loss_csv.to_csv(self.loss_file_path, mode='a', header=False, index=False)
            if epoch % 500 == 0:
                print('Epoch {:03d} train_loss: {:.4f} test_loss:{:.4f}'.format(
                    epoch, loss.item(), test_loss.item()))
                torch.save(model.state_dict(), self.train_process + '/epoch_{}.pth'.format(epoch))

        torch.save(model.state_dict(), self.model_file_path)
        writer.close()

    def test(self):
        if self.model == 'gcn':
            print('gcn')
            test_model = GCN(self.class_num, self.program)

        test_model.load_state_dict(torch.load(self.model_file_path))
        test_model.to(self.device)
        test_model.eval()
        h, out, _ = test_model(self.data, self.K, self.device)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[self.data.test_mask] == self.data.y[
            self.data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())  # Derive ratio of correct predictions.
        print('标签预测精度：{}'.format(test_acc))
        return test_acc, h

    def covariance_caculate(self, h, node_mapping):
        h = h.cpu()
        co = h @ h.T

        co_process = process_covariance(co)

        values, indices = torch.topk(co_process.flatten(), k=20, dim=0)
        node_row = []
        node_col = []
        for i in range(20):
            node_row.append(int((indices[i] // co.shape[0]).item()))
            node_col.append(int((indices[i] % co.shape[0]).item()))
            print(
                f'Row index of {i + 1}th largest element: {indices[i] // co.shape[0]}, column index: {indices[i] % co.shape[0]}')

        df = pd.read_csv("./data/nodeidx2paperid.csv", header=0)
        paper_mapping = dict(zip(df.values[:, 0].tolist(), df.values[:, [1]]))

        titleabs = pd.read_table("./data/titleabs.tsv", header=None)
        title_mapping = dict(zip(tuple(titleabs.values[:, 0].tolist()), tuple(titleabs.values[:, [1, 2]].tolist())))

        for i in range(20):
            title1 = title_mapping[paper_mapping[node_mapping[node_row[i]]].item()][0]
            abstract1 = title_mapping[paper_mapping[node_mapping[node_row[i]]].item()][1]
            title2 = title_mapping[paper_mapping[node_mapping[node_col[i]]].item()][0]
            abstract2 = title_mapping[paper_mapping[node_mapping[node_col[i]]].item()][1]
            print(
                f'第 {i + 1}大元素:\n第1篇论文标题是 《{title1}》, 摘要是 {abstract1}；\n 第2篇论文标题是 《{title2}》, 摘要是 {abstract2}')
            print(self.data.y[node_row[i]])
            print(self.data.y[node_col[i]])
            print(
                '==================================================================================================================================================')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='PLS-GNN')
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'cheb'])
    parser.add_argument('--K', type=int, default='3', help='The number of iterations of the PLS')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epoch', type=int, default=3000, help="epoch")
    parser.add_argument('--seed', type=int, default=12345, help="Random seed")
    parser.add_argument('--weight', type=float, default=0.1, help='weight')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu is selected')
    parser.add_argument('--program', type=str, default='tanh', choices=['tanh+none', 'tanh', 'leakyrule'])
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    numpy.random.seed(SEED)
    # 需要调节的参数
    parameter = {'lr': args.lr, 'epoch': args.epoch, 'weight': args.weight}

    with open('./data/graph_data+node_mapping(select_label_4).pkl', 'rb') as f:
        data = pickle.load(f)
        node_mapping = pickle.load(f)
    data.x = data.x.tanh()
    pls = PLS_model(
        data=data,
        device=device,
        optimizer=args.optimizer,
        model=args.model,
        K=args.K,
        program=args.program,
        para=parameter
    )
    pls.train()  # step-1
    test_acc, h = pls.test()  # step-2
    pls.covariance_caculate(h, node_mapping)  # step-3
