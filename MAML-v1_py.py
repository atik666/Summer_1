import torch
import numpy as np
import os
import zipfile
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy,copy

root_dir = '/home/atik/Documents/MAML/Summer_1/datasets/python'

img_list = np.load(os.path.join(root_dir, 'omniglot.npy')) # (1623, 20, 1, 28, 28)
x_train = img_list[:1200]
x_test = img_list[1200:]
num_classes = img_list.shape[0]
datasets = {'train': x_train, 'test': x_test}

n_way = 5
k_spt = 1  ## support data 
k_query = 15 ## query data 
imgsz = 28
resize = imgsz
task_num = 8
batch_size = task_num

indexes = {"train": 0, "test": 0}
datasets = {"train": x_train, "test": x_test}
print("DB: train", x_train.shape, "test", x_test.shape)

setsz = k_spt * n_way
querysz = k_query * n_way
data_cache = []

def load_data_cache(dataset):
    """
    Collects several batches data for N-shot learning
    :param dataset: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(10):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace =  False) 

            for j, cur_class in enumerate(selected_cls):

                selected_img = np.random.choice(20, k_spt + k_query, replace = False)

                # support & query
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]
 
            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

#         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int).reshape(batch_size, querysz)
#         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache

datasets_cache = {"train": load_data_cache(x_train),  # current epoch data cached
                       "test": load_data_cache(x_test)}


def next(mode='train'):
    """
    Gets next batch from the dataset with name.
    :param mode: The name of the splitting (one of "train", "val", "test")
    :return:
    """
    # update cache if indexes is larger than len(data_cache)
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.vars = nn.ParameterList()  # tensor
        self.vars_bn = nn.ParameterList()
        
        # conv2d
        # in_channels = 1, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2
        weight = nn.Parameter(torch.ones(64, 1, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # BatchNorm
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # conv2d
        # in_channels = 1, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # BatchNorm
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # conv2d
        # in_channels = 1, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # BatchNorm
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # conv2d
        # in_channels = 1, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        # BatchNorm
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight,bias])
        
        running_mean = nn.Parameter(torch.zeros(64), requires_grad= False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad= False)
        self.vars_bn.extend([running_mean, running_var])
        
        # linear
        weight = nn.Parameter(torch.ones([5,64]))
        bias = nn.Parameter(torch.zeros(5))
        self.vars.extend([weight,bias])
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2), 
            
#             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 2, stride = 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2), 
            
#             nn.Flatten(),
#             nn.Linear(64,5)
#         )        
 
        
    def forward(self, x, params = None, bn_training=True):
        '''
        :bn_training: set False to not update
        :return: 
        '''
        if params is None:
            params = self.vars
        
        weight, bias = params[0], params[1]  # CONV
        x = F.conv2d(x, weight, bias, stride = 2, padding = 2)
        
        weight, bias = params[2], params[3]  # BN
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.max_pool2d(x,kernel_size=2)  # MAX_POOL 
        x = F.relu(x, inplace = [True])  # relu
        
        weight, bias = params[4], params[5]  # CONV
        x = F.conv2d(x, weight, bias, stride = 2, padding = 2)
        
        weight, bias = params[6], params[7]  # BN
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.max_pool2d(x,kernel_size=2)  # MAX_POOL  
        x = F.relu(x, inplace = [True])  # relu
        
        weight, bias = params[8], params[9]  # CONV
        x = F.conv2d(x, weight, bias, stride = 2, padding = 2)
        
        weight, bias = params[10], params[11]  # BN
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.max_pool2d(x,kernel_size=2)  # MAX_POOL
        x = F.relu(x, inplace = [True])  # relu
        
        weight, bias = params[12], params[13]  # CONV
        x = F.conv2d(x, weight, bias, stride = 2, padding = 2)
        x = F.relu(x, inplace = [True])  # relu
        weight, bias = params[14], params[15]  # BN
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x = F.batch_norm(x, running_mean, running_var, weight=weight,bias =bias, training= bn_training)
        x = F.max_pool2d(x,kernel_size=2)  # MAX_POOL
        
        x = x.view(x.size(0), -1) ## flatten
        weight, bias = params[16], params[17]  # linear
        x = F.linear(x, weight, bias)
        
        output = x
        
        return output
    
    
    def parameters(self):
        
        return self.vars


class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.update_step = 5 # task-level inner update steps
        self.update_step_test = 5  
        self.net = BaseNet()
        self.meta_lr = 2e-4
        self.base_lr = 4 * 1e-2
        self.inner_lr = 0.4
        self.outer_lr = 1e-2
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)
        
    def forward(self,x_spt, y_spt, x_qry, y_qry):
        # initialization
        task_num, ways, shots, h, w = x_spt.size()
        query_size = x_qry.size(1) # 75 = 15 * 5
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]
        
        for i in range(task_num):
            ## Step initial Update
            y_hat = self.net(x_spt[i], params = None, bn_training=True) # (ways * shots, ways)
            loss = F.cross_entropy(y_hat, y_spt[i]) 
            grad = torch.autograd.grad(loss, self.net.parameters())
            tuples = zip(grad, self.net.parameters()) ## Combine the gradient and parameters\theta
            # fast_weights update: theta - alpha*nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # query - calculate accuracy
            # This step uses the data before the update
            with torch.no_grad():
                y_hat = self.net(x_qry[i], self.net.parameters(), bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[0] += correct
            
            # Test on the query set with the updated data
            with torch.no_grad():
                y_hat = self.net(x_qry[i], fast_weights, bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[1] += correct   
            
            for k in range(1, self.update_step):
                
                y_hat = self.net(x_spt[i], params = fast_weights, bn_training=True)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights) 
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                    
                y_hat = self.net(x_qry[i], params = fast_weights, bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[k+1] += loss_qry
                
                with torch.no_grad():
                    pred_qry = F.softmax(y_hat,dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    correct_list[k+1] += correct
                
        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad() # Gradient clear
        loss_qry.backward()
        self.meta_optim.step()

        # Convert to numpy array
        loss_list_qry = torch.stack(loss_list_qry)
        loss_list_qry = loss_list_qry.cpu().detach().numpy()
        
        accs = np.array(correct_list) / (query_size * task_num)
        loss = np.array(loss_list_qry) / ( task_num)
        return accs,loss

    
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4
        
        query_size = x_qry.size(0)
        correct_list = [0 for _ in range(self.update_step_test + 1)]
        
        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))
        
        # Test on the query set and calculate the accuracy
        # This step uses the data before the update
        with torch.no_grad():
            y_hat = new_net(x_qry,  params = new_net.parameters(), bn_training = True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[0] += correct

        # Test on the query set with the updated data
        with torch.no_grad():
            y_hat = new_net(x_qry, params = fast_weights, bn_training = True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params = fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
            
            y_hat = new_net(x_qry, fast_weights, bn_training=True)
            
            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry).sum().item()
                correct_list[k+1] += correct
                
        del new_net
        accs = np.array(correct_list) / query_size
        return accs


## omniglot

import time
from tqdm import tqdm

device = torch.device('cuda')

meta = MetaLearner().to(device)

epochs = 60001
for step in tqdm(range(epochs)):
    start = time.time()
    x_spt, y_spt, x_qry, y_qry = next('train')
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device),\
                                 torch.from_numpy(y_spt).to(device),\
                                 torch.from_numpy(x_qry).to(device),\
                                 torch.from_numpy(y_qry).to(device)
    accs,loss = meta(x_spt, y_spt, x_qry, y_qry)
    end = time.time()
    if step % 100 == 0:
        print("epoch:" , step)
        print("acc:" , accs)
        print("loss:" , loss)
        
    if step % 1000 == 0:
        accs = []
        for _ in range(1000//task_num):
            # db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device),\
                                         torch.from_numpy(y_spt).to(device),\
                                         torch.from_numpy(x_qry).to(device),\
                                         torch.from_numpy(y_qry).to(device)

            
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)
                
        print("\n")
        print('before the mean process：',np.array(accs).shape)
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('test set accuracy:',accs)
                                            







