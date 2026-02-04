import os
import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score

from options import get_dataloader, get_config
from data import *
from loss_function import get_loss
from model import net
# np.set_printoptions(precision=4, suppress=True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train_test(args):
    device = args.device

    config, dataloader = get_config(args.dataset) 
    train_loader, test_loader = get_dataloader(args.dataset, args.conflictive_test) 

    # 初始化模型 (model.py 里的 net 类会自动适应不同数据集的维度)
    model = net(dataloader.num_views, num_layer = config['layer_num'], dims=dataloader.dims, num_classes=dataloader.num_classes)
    model.to(device)
    # print(model) # 如果嫌刷屏太长，可以注释掉这行

    # 优化器会同时优化 FUML 和 DualVAE 的参数
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    best_test_acc = 0
    best_test_F1 = 0
    best_test_precision = 0
    
    # -------------------------------------------------------------------------
    # [关键策略] Beta Warm-up 参数设置
    # -------------------------------------------------------------------------
    warmup_epochs = 100  # 前 100 轮属于预热期，让 FUML 先把底子打好
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        
        # ---------------------------------------------------------------------
        # [动态调整] 计算当前轮次的 Beta
        # ---------------------------------------------------------------------
        if epoch <= warmup_epochs:
            # 线性增长：第1轮接近0，第100轮达到设定的 args.beta
            # 作用：在 VAE 还没学会修图时，屏蔽它的噪声干扰
            current_beta = args.beta * (epoch / warmup_epochs)
        else:
            # 预热结束，保持恒定，让 VAE 全力辅助
            current_beta = args.beta
        # ---------------------------------------------------------------------

        total_vae_loss = 0
        total_cls_loss = 0
        
        for X, Y, indexes in train_loader:
            for v in range(dataloader.num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)

            # --- [关键] 接收 4 个返回值 ---
            # vae_loss 是 DualVAE 计算出的 (重构误差 + KL散度)
            Credibility, MMcrediblity, MMuncertainty, vae_loss = model(X, Y, test=False)
            
            # 计算分类损失 (FUML 原有逻辑)
            cls_loss = get_loss(Credibility, MMcrediblity, Y, dataloader.num_classes)
            
            # --- [关键] 使用动态 Beta 融合 Loss ---
            loss = cls_loss + current_beta * vae_loss
            
            optimizer.zero_grad()
            loss.backward()

            # --- [关键] 梯度裁剪 ---
            # 防止梯度爆炸导致 Acc 突然归零，max_norm=5.0 是经验值
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_vae_loss += vae_loss.item()
            total_cls_loss += cls_loss.item()
        
        # 每 10 轮测试一次
        if epoch % 10 == 0:
            model.eval()
            num_correct, num_sample = 0, 0

            Y_pre_total = None
            Y_total = None
            MMuncertainty_total = None

            for X, Y, indexes in test_loader:
                for v in range(dataloader.num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)

                with torch.no_grad():
                    # 测试阶段，test=True，vae_loss 返回 0
                    Credibility, MMcrediblity, MMuncertainty, _ = model(X, test=True)
                    
                    _, Y_pre = torch.max(MMcrediblity, dim=1)
                    num_correct += (Y_pre == Y).sum().item()
                    num_sample += Y.shape[0]
                
                Y_pre = np.array(Y_pre.cpu())    
                Y = np.array(Y.cpu())  
                MMcrediblity = np.array(MMcrediblity.cpu())
                MMuncertainty = np.array(MMuncertainty.cpu()) 

                if Y_pre_total is None:
                    Y_pre_total = Y_pre
                    Y_total = Y
                    MMuncertainty_total = MMuncertainty
                else:
                    Y_pre_total = np.hstack([Y_pre_total, Y_pre])
                    Y_total = np.hstack([Y_total, Y])
                    MMuncertainty_total = np.hstack([MMuncertainty_total.squeeze(), MMuncertainty.squeeze()])
            
            acc = num_correct / num_sample
            F1 = f1_score(Y_total, Y_pre_total, average='macro')
            precision = precision_score(Y_total, Y_pre_total, average='macro')
            
            if acc > best_test_acc:
                best_test_acc = acc
                best_test_F1 = F1
                best_test_precision = precision

            # 打印日志：带上当前的 Beta 值，方便确认 Warm-up 是否生效
            print('Epoch:{:.0f} (Beta:{:.5f}) ====> best acc: {:.4f} acc: {:.4f} F1: {:.4f}  P: {:.4f} uncer: {:.4f}'.format(
                epoch, current_beta, best_test_acc, acc, F1, precision, np.mean(MMuncertainty_total)))
            
    return best_test_acc, best_test_precision, best_test_F1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PIE', metavar='N',
                        help='dataset name') # PIE, Scene, LandUse, HW, NUSOBJ, Fashion, Leaves, MSRC
    parser.add_argument('--conflictive_test', type=bool, default=False, metavar='N',
                        help='conflicting or not')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='N',
                        help='gpu or cpu')
    parser.add_argument('--seed_list', type=int, default=[1,2,3,4,5,6,7,8,9,10], metavar='N',
                        help='random seed') 
    
    # 增加 beta 参数接口，方便命令行调节
    parser.add_argument('--beta', type=float, default=0.005, metavar='N',
                        help='weight for VAE reconstruction loss')
    
    args = parser.parse_args()

    print("Processor: ", os.getpid())
    print(f"Start Training on {args.dataset} with Target Beta = {args.beta} (Warm-up enabled)")
    
    Acc_list = []
    P_list = []
    F_score_list = []
    
    for seed in args.seed_list:
        print("\n" + "="*40)
        print("seed = ", seed)
        print("="*40)
        setup_seed(seed)
        Acc,P,F_score = train_test(args)
        Acc_list.append(Acc)
        P_list.append(P)
        F_score_list.append(F_score)

    print("\n***************************************")
    print("************ FINAL RESULTS ************")
    print("***************************************")
    print("Dataset :", args.dataset)
    print("Acc     :", str(round(np.mean(Acc_list), 4)), " +- ", str(round(np.std(Acc_list), 4)) )
    print("P       :", str(round(np.mean(P_list), 4)), " +- ", str(round(np.std(P_list), 4)) )
    print("F_score :", str(round(np.mean(F_score_list), 4)), " +- ", str(round(np.std(F_score_list), 4)) )
    print("***************************************")
    print("***************************************")