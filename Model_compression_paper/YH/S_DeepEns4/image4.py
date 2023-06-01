import argparse
from datetime import date
import os
import numpy as np
import copy
from datetime import datetime
from progress.bar import ChargingBar as Bar
import queue
import math

import sys 
sys.path.append('..')

from utils_IS import *
from evaluate import *
from S_DeepEns2.models import *

parser = argparse.ArgumentParser(description='PyTorch sparse DeepEnsemble using LeGR')

########################## model setting ##########################
parser.add_argument('--model', type=str, default= 'resnet18', choices=['MLP', 'resnet18'], help='architecture of model')
parser.add_argument('--L', type=int, default= 2, help='depth of MLP')
parser.add_argument('--p', type=int, default= 100, help='width of MLP')


########################## basic setting ##########################
parser.add_argument('--start_seed', type=int, default=0, help='start_seed')
parser.add_argument('--end_seed', type=int, default=2, help='end_seed')
parser.add_argument('--gpu', default=2, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='/home/ggong369/data/MBNN_ICML2023/datasets', help='Directory of dataset')
parser.add_argument('--out', default='/home/ggong369/data/MBNN_ICML2023/experiments', help='Directory to output the result')


######################### Dataset setting #############################
parser.add_argument('--dataset', type=str, default= 'CIFAR10', choices=['Boston', 'Concrete', 'Energy', 'Yacht', 'CIFAR10', 'CIFAR100'], help='benchmark dataset')
parser.add_argument('--batch_size', default=100, type=int, help='train batchsize')

######################### Targeted dropout setting #############################
parser.add_argument('--target', default=0.6, type=float, help='target node sparsity') # 1.0
parser.add_argument('--POP', default=16, type=int, help='POPULATIONS')
parser.add_argument('--SAM', default=4, type=int, help='SAMPLES')
parser.add_argument('--GEN', default=50, type=int, help='GENERATIONS') #50
parser.add_argument('--SS', default=0.1, type=float, help='SCALE_SIGMA')
parser.add_argument('--MP', default=0.1, type=float, help='MUTATE_PERCENT')

######################### MC setting #############################
parser.add_argument('--num_sample', default=5, type=int, help='the number of MC sample')
parser.add_argument('--tau_hat', default=200, type=int, help='The number of updates before evaluating for fitness (used in EA).')
parser.add_argument('--long_ft', default=60, type=int, help='It specifies how many epochs to fine-tune the network once the pruning is done')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

######################### add name #############################
parser.add_argument('--add_name', default='', type=str, help='add_name')


args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():
    
    out_directory = args.out + '/DeepEns_LeGR' + '/' + str(args.dataset)
    out_directory += '/' + str(date.today().strftime('%Y%m%d')[2:])
        
    out_directory += '/target' + str(args.target) + '_GEN' + str(args.GEN) + '_SS' + str(args.SS)
    out_directory += '_long_ft' + str(args.long_ft) + '_lr' + str(args.lr) + 'bn'

    if args.add_name != '':
        out_directory +='_'+str(args.add_name)

    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
        
        
    result1_list = []
    result2_list = []
    result3_list = []
    result4_list = []
    result5_list = []    
    
    for seed in range(args.start_seed, args.end_seed+1):           
        print("seed : ", seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        
        #dataset
        trainloader, testloader, n_train, n_test, p_data, num_classes = data_process(args.dataset, args.data_dir, seed, args.batch_size)
        
        task = 'classification'
        criterion = torch.nn.CrossEntropyLoss()
        
        finetuning_epoch = int(args.tau_hat/len(trainloader)) + 1

        for mt in range(args.num_sample):
            model = M_ResNet18(num_classes).cuda()
            
            n_h_nodes = [] 
            for name, param in model.named_parameters():
                if 'M_relu' in name:
                    n_h_nodes.append(int(param.sum().item()))
                        
            model.load_state_dict(torch.load(args.out + '/pretrain/' + args.dataset + '/221128/epoch100_lr0.001/seed_'+str(seed)+'_mt_'+str(mt)+'.pt', map_location='cuda:'+str(args.gpu)))
            
            original_dist=[filter_ranks(model.conv1, model.bn1)]      
            for layer in model.modules():
                if isinstance(layer, M_BasicBlock):
                    original_dist = original_dist + layer.get_2norm()
                    
            original_dist_stat = {}
            for k, stat in enumerate(original_dist):
                a = stat.detach().cpu().numpy()
                original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}
                
            index_queue = queue.Queue(args.POP)
            population_loss = np.zeros(0)
            population_data = []
            minimum_loss = 9999
                       
            # evoluation algorithm
            bar = Bar('{:>12}'.format('evoluation for mt '+ str(mt)), max=args.GEN)
            for i in range(args.GEN):
                step_size = 1-(float(i)/(args.GEN*1.25))
                perturbation = []

                if i == args.POP-1:
                    for _ in range(len(original_dist_stat)):
                        perturbation.append((1,0))
                elif i < args.POP-1:   
                    for k in range(len(original_dist_stat)):                        
                        scale = np.exp(float(np.random.normal(0, args.SS)))
                        shift = float(np.random.normal(0, original_dist_stat[k]['std'])) - original_dist_stat[k]['mean']
                        perturbation.append((scale, shift))
                else:
                    sampled_idx = np.random.choice(args.POP, args.SAM)
                    sampled_loss = population_loss[sampled_idx]
                    winner_idx_ = np.argmin(sampled_loss)
                    winner_idx = sampled_idx[winner_idx_]
                    oldest_index = index_queue.get()        
                    base = population_data[winner_idx]
                    mnum = math.ceil(args.MP * len(original_dist_stat))
                    mutate_candidate = np.random.choice(len(original_dist_stat), mnum)
                    for k in range(len(original_dist_stat)):
                        scale = 1
                        shift = 0
                        if k in mutate_candidate:
                            scale = np.exp(float(np.random.normal(0, args.SS*step_size)))
                            shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                        perturbation.append((scale*base[k][0], shift+base[k][1]))
                        
                        
                model_tmp = copy.deepcopy(model)
                Masking_layers=[]
                for name, param in model_tmp.named_parameters():
                    if 'M_relu' in name:
                        Masking_layers.append(param)

                perturbed_dist = []
                for k in range(len(original_dist_stat)):
                    perturbed_dist.append(original_dist[k] * perturbation[k][0] + perturbation[k][1])

                cutoff = torch.quantile(torch.hstack(perturbed_dist), 1 - args.target)

                for k in range(len(original_dist_stat)):
                    idx = torch.where(perturbed_dist[k] < cutoff)[0]
                    Masking_layers[k].data[idx] *= 0
                    if len(idx)==len(Masking_layers[k]):
                        idx = torch.argmax(perturbed_dist[k])
                        Masking_layers[k].data[idx] += 1

                optimizer = torch.optim.Adam(model_tmp.parameters(), lr=args.lr)
                for epoch in range(finetuning_epoch):
                    train(model_tmp, task, trainloader, criterion, optimizer)

                ERROR_train = test(model_tmp, task, trainloader)

                if ERROR_train.item() < minimum_loss:
                    minimum_loss = ERROR_train.item()
                    best_perturbation = perturbation
                    
                    ERROR_test = test(model_tmp, task, testloader)
                    bar.suffix  = ' Gen : {gen} | ERROR_train : {ER_tr} | ERROR_test : {ER_te})'.format(
                                    gen = i,
                                    ER_tr=ERROR_train.item(),
                                    ER_te=ERROR_test.item()
                                    )

                if i < args.POP:
                    index_queue.put(i)
                    population_data.append(perturbation)
                    population_loss = np.append(population_loss, ERROR_train.item())
                else:
                    index_queue.put(oldest_index)
                    population_data[oldest_index] = perturbation
                    population_loss[oldest_index] = ERROR_train.item()
                    
                bar.next()
            bar.finish()
            
            model_tmp = copy.deepcopy(model)
            
            Masking_layers=[]
            for name, param in model_tmp.named_parameters():
                if 'M_relu' in name:
                    Masking_layers.append(param)

            perturbed_dist = []
            for k in range(len(original_dist_stat)):
                perturbed_dist.append(original_dist[k] * best_perturbation[k][0] + best_perturbation[k][1])

            cutoff = torch.quantile(torch.hstack(perturbed_dist), 1 - args.target)

            for k in range(len(original_dist_stat)):
                idx = torch.where(perturbed_dist[k] < cutoff)[0]
                Masking_layers[k].data[idx] *= 0
                if len(idx)==len(Masking_layers[k]):
                    idx = torch.argmax(perturbed_dist[k])
                    Masking_layers[k].data[idx] += 1

            optimizer = torch.optim.Adam(model_tmp.parameters(), lr=args.lr)
            for epoch in range(args.long_ft):
                train(model_tmp, task, trainloader, criterion, optimizer)

            ERROR_train = test(model_tmp, task, trainloader)
            ERROR_test = test(model_tmp, task, testloader)
            
            print(" mt : ", mt, " train error : ", ERROR_train.item(), " test error : ", ERROR_test.item())
            torch.save(model_tmp.state_dict(), out_directory + '/seed_%d_mt_%d.pt'%(seed,mt))
            
            n_act_h_nodes = []
            for name, param in model_tmp.named_parameters():
                if 'M_relu' in name:
                    n_act_h_nodes.append(int(param.sum().item()))      
            n_act_i_nodes = p_data

            macs, params = profiling(args.model, p_data, n_act_i_nodes, num_classes, n_act_h_nodes, n_h_nodes=n_h_nodes)
            print(100 * (macs))
            
        
        # test #######################################
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
            result1_list.append(ACC)
            result2_list.append(m_NLL)
            result3_list.append(ECE)
            result4_list.append(100 * (macs))
            result5_list.append(100 * (params))
    
    num_seed = args.end_seed - args.start_seed
    result1_list, result2_list, result3_list, result4_list, result5_list = np.stack(result1_list), np.stack(result2_list), np.stack(result3_list), np.stack(result4_list), np.stack(result5_list)
    print("%.3f(%.3f), %.3f(%.3f), %.3f(%.3f), %.2f(%.2f), %.2f(%.2f)" % (np.mean(result1_list), np.std(result1_list)/np.sqrt(num_seed),np.mean(result2_list), np.std(result2_list)/np.sqrt(num_seed),np.mean(result3_list), np.std(result3_list)/np.sqrt(num_seed),np.mean(result4_list), np.std(result4_list)/np.sqrt(num_seed),np.mean(result5_list), np.std(result5_list)/np.sqrt(num_seed)))   
   
    
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