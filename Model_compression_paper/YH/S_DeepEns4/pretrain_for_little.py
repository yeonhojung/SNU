import argparse
from datetime import date
import os
import numpy as np
import copy
from datetime import datetime
from progress.bar import ChargingBar as Bar

import sys 
sys.path.append('..')

from utils_IS import *
from evaluate import *
from S_DeepEns.models import *


parser = argparse.ArgumentParser(description='PyTorch BNN')

########################## model setting ##########################
parser.add_argument('--model', type=str, default= 'MLP', choices=['MLP', 'resnet18'], help='architecture of model')
parser.add_argument('--L', type=int, default= 2, help='depth of MLP')
parser.add_argument('--p', type=int, default= 100, help='width of MLP')


########################## basic setting ##########################
parser.add_argument('--start_seed', type=int, default=0, help='start_seed')
parser.add_argument('--end_seed', type=int, default=19, help='end_seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='/home/ggong369/data/MBNN_ICML2023/datasets', help='Directory of dataset')
parser.add_argument('--out', default='/home/ggong369/data/MBNN_ICML2023/experiments', help='Directory to output the result')


######################### Dataset setting #############################
parser.add_argument('--dataset', type=str, default= 'Boston', choices=['Boston', 'Concrete', 'Energy', 'Yacht', 'CIFAR10', 'CIFAR100'], help='benchmark dataset')
parser.add_argument('--batch_size', default=100, type=int, help='train batchsize')


######################### MCMC setting #############################
parser.add_argument('--num_sample', default=5, type=int, help='the number of MCMC sample')
parser.add_argument('--total_epoch', default=400, type=int, help='total epoch for each model')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

######################### add name #############################
parser.add_argument('--add_name', default='', type=str, help='add_name')



parser.add_argument('--rate', default=0.7, type=float, help='rrrr')


args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)


def main():
    
    out_directory = args.out + '/pretrain' + '/' + str(args.dataset)
    out_directory += '/' + str(date.today().strftime('%Y%m%d')[2:])
    
    out_directory += '/epoch' + str(args.total_epoch) + '_lr' + str(args.lr) + 'little' + str(args.rate)

    if args.add_name != '':
        out_directory +='_'+str(args.add_name)

    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
    
    for seed in range(args.start_seed, args.end_seed+1):
        str_time = datetime.now()
        print("seed : ", seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
            
        #dataset
        trainloader, testloader, n_train, n_test, p_data, num_classes = data_process(args.dataset, args.data_dir, seed, args.batch_size)
            
        #task
        if num_classes==1:
            task = 'regression'
            criterion = torch.nn.MSELoss()
        else:
            task = 'classification'
            criterion = torch.nn.CrossEntropyLoss()
            
        for mt in range(args.num_sample):
            
            #model
            if args.model == 'MLP':
                model = M_MLP(p_data, num_classes, args.L, args.p).cuda()
            elif args.model == 'resnet18':
                model = M_ResNet18(num_classes).cuda()
                
                for name, param in model.named_parameters():
                    if 'M_relu' in name:
                        param.data[int(args.rate*len(param)):] *= 0
            
            bar = Bar('{:>12}'.format('{mt}th model Training'.format(mt=mt+1)), max=args.total_epoch)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(args.total_epoch):

                train(model, task, trainloader, criterion, optimizer)

                if (epoch+1)%(args.total_epoch/10)==0:
                    ERROR_train = test(model, task, trainloader)
                    ERROR_test = test(model, task, testloader)

                    bar.suffix  = '({epo}/{total_epo}) ERROR_train : {ER_tr} | ERROR_test : {ER_te})'.format(
                                    epo=epoch + 1,
                                    total_epo=args.total_epoch,
                                    ER_tr=ERROR_train.item(),
                                    ER_te=ERROR_test.item()
                                    )
                bar.next()
            
            if task == 'regression':
                model.sigma.data = ERROR_train.reshape([-1])
            bar.finish()
            torch.save(model.state_dict(), out_directory + '/seed_%d_mt_%d.pt'%(seed, mt))
            
        print("model testing")
        pred_list=[]
        target_list=[]
        sigma_list=[]
        macs_list=[]
        params_list=[]
        
        with torch.no_grad():
            for mt in range(args.num_sample):
                model = M_ResNet18(num_classes).cuda()
                model.eval()
                
                n_h_nodes = [] 
                for name, param in model.named_parameters():
                    if 'M_relu' in name:
                        n_h_nodes.append(int(param.sum().item()))
                
                model.load_state_dict(torch.load(out_directory + '/seed_%d_mt_%d.pt'%(seed,mt), map_location='cuda:'+str(args.gpu)))
                pred = []
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    pred.append(F.softmax(outputs,dim=1))              

                    if mt==0:
                        target_list.append(targets.squeeze())

                pred_list.append(torch.cat(pred,0))                            
                                       
                n_act_h_nodes = []
                for name, param in model.named_parameters():
                    if 'M_relu' in name:
                        n_act_h_nodes.append(int(param.sum().item()))      
                n_act_i_nodes = p_data

                macs, params = profiling(args.model, p_data, n_act_i_nodes, num_classes, n_act_h_nodes, n_h_nodes=n_h_nodes)
                macs_list.append(macs)
                params_list.append(params) 
                
                
            pred_list = torch.stack(pred_list)
            target_list = torch.cat(target_list,0)           
            ACC, m_NLL, ECE = evaluate_averaged_model_classification(pred_list, target_list)
            
            
            macs = np.stack(macs_list).mean()
            params = np.stack(params_list).mean()
            print("ACC : ", ACC, " m_NLL : ", m_NLL, " ECE : ", ECE, "FLOPs rate : ", 100 * (macs), " non-zero param rate : ", 100 * (params))            
            

            
        end_time = datetime.now()
        time_delta = end_time - str_time
        print("Seed ", seed, " done, run-time : ", time_delta)
            
def train(model, task, dataloader, criterion, optimizer):
    model.train()
    if task == 'regression':        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    elif task == 'classification':
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda().squeeze().long()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()        
            
def test(model, task, dataloader):
    ERROR = 0
    model.eval()
    with torch.no_grad():
        if task == 'regression':        
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                ERROR += ((targets - outputs)**2).sum()
            return torch.sqrt(ERROR/len(dataloader.dataset))
        
        elif task == 'classification':
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda().squeeze().long()
                outputs = model(inputs)
                ERROR += (torch.argmax(outputs,1) != targets).sum()    
            return ERROR/len(dataloader.dataset)

                    
if __name__ == '__main__':
    main()
