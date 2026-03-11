# train.py
import os
import copy
import argparse
from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataload_KIDA import CustomVideoDataset
from models.KIDA_net import MultiStageModel
from logger import CompleteLogger
from utils.metrics import calculate_metrics
from models.IEA_loss import IEALoss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    import numpy as np, random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    set_seed(args.seed)
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Paths
    root_data_path = args.data_path
    data_split_train_path = root_data_path + "/train_emb_" + args.emb_path + "/"
    data_split_test_path = root_data_path + "/test_emb_" + args.emb_path + "/"
    flow_dir_train = "./data/flow6d_train"
    flow_dir_test = "./data/flow6d_test"
    text_emb = torch.load('./data/text_emb/error_text_emb_p.pt', map_location='cpu')

    # Dataloaders
    train_dataset = CustomVideoDataset(data_split_train_path, flow_dir_train, text_emb)
    test_dataset = CustomVideoDataset(data_split_test_path, flow_dir_test, text_emb)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.work, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.work, worker_init_fn=worker_init_fn)

    # Model & Optimizer
    model = MultiStageModel(args.num_block, args.emb_type, args.com_factor, args.text_dim, args.feature_dim, args.num_class).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_test_AUC, best_test_mAP, best_epoch = 0.0, 0.0, 0
    save_dir = f"./ExpLog/{args.exp}/{args.emb_type}/{args.lr}/{args.seed}/{formatted_time}/"
    os.makedirs(save_dir, exist_ok=True)
    logger = CompleteLogger(save_dir)

    for epoch in range(args.epoch):
        # --- Train ---
        model.train()
        train_loss = 0.0
        iea_criterion = IEALoss(err_weight=1.25).to(device)
        train_scores, train_labels = [], []

        for data in train_dataloader:
            optimizer.zero_grad()
            video_fe, text_pfe, vl, e_labels, flow_mag = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device).squeeze(0), data[4].to(device)
            video_fe = video_fe.transpose(2, 1)

            predictions, delta, q_p = model.forward(video_fe, text_pfe, flow_mag)
            predictions = predictions.squeeze()
            
            # Loss Calculation
            loss_cls = criterion(predictions, e_labels.float())
            loss_align = iea_criterion(delta, q_p, e_labels)
            loss = loss_cls + args.lambda_align * loss_align

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_scores.extend(torch.sigmoid(predictions).flatten().tolist())
            train_labels.extend(e_labels.flatten().tolist())

        train_AUC, train_mAP = calculate_metrics(train_labels, train_scores)

        # --- Evaluate ---
        model.eval()
        test_scores, test_labels = [], []
        with torch.no_grad():
            for data in test_dataloader:
                video_fe, text_pfe, vl, e_labels, flow_mag = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device).squeeze(0), data[4].to(device)
                video_fe = video_fe.transpose(2, 1)

                predictions, _, _ = model.forward(video_fe, text_pfe, flow_mag)
                test_scores.extend(torch.sigmoid(predictions.squeeze()).flatten().tolist())
                test_labels.extend(e_labels.flatten().tolist())
                
        test_AUC, test_mAP = calculate_metrics(test_labels, test_scores)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_dataloader):.3f} | Train AUC: {train_AUC*100:.2f}% | Test AUC: {test_AUC*100:.2f}% | Test mAP: {test_mAP*100:.2f}%")

        # Save Best Model
        if test_AUC > best_test_AUC or (test_AUC == best_test_AUC and test_mAP > best_test_mAP):
            best_test_AUC, best_test_mAP, best_epoch = test_AUC, test_mAP, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_weight.pth"))
            print(f"Updated best model at Epoch {epoch} with AUC: {best_test_AUC*100:.2f}%")

    print(f"\nTraining Complete Best Epoch: {best_epoch} | Best AUC: {best_test_AUC*100:.2f}% | Best mAP: {best_test_mAP*100:.2f}%")
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", default="KIDA_SAR", type=str)
    parser.add_argument("-lb", "--lambda_align", default=0.1, type=float)
    parser.add_argument("-dp", "--data_path", default="/memory/luoyuxuan4090/SEDMamba/data", type=str)
    parser.add_argument("-gpu_id", default="cuda:0", type=str)
    parser.add_argument("-w", "--work", default=4, type=int)
    parser.add_argument("-s", "--seed", default=3407, type=int)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("-e", "--epoch", default=120, type=int)
    parser.add_argument("-l", "--lr", default=2e-5, type=float)
    parser.add_argument("-et","--emb_type", default='dinoemb', type=str)
    parser.add_argument("-ep","--emb_path", default='DINOv2', type=str)
    parser.add_argument("-cls", "--num_class", default=1, type=int)
    parser.add_argument("-fd", "--text_dim", default=768, type=int)
    parser.add_argument("-id", "--feature_dim", default=1000, type=int)
    parser.add_argument("-nb", "--num_block", default=3, type=int)
    parser.add_argument("-g", "--com_factor", default=64, type=int)
    main(parser.parse_args())