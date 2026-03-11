# test.py
import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader

from dataset.dataload_KIDA import CustomVideoDataset
from models.KIDA_net import MultiStageModel
from utils.metrics import calculate_metrics

def main(args):
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    
    # Paths
    data_split_test_path = args.data_path + "/test_emb_" + args.emb_path + "/"
    flow_dir_test = "./data/flow6d_test"
    text_emb = torch.load('./data/text_emb/error_text_emb_p.pt', map_location='cpu')

    # Dataloader
    test_dataset = CustomVideoDataset(data_split_test_path, flow_dir_test, text_emb)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.work)

    # Load Model
    model = MultiStageModel(args.num_block, args.emb_type, args.com_factor, args.text_dim, args.feature_dim, args.num_class).to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.eval()

    test_scores, test_preds, test_labels = [], [], []
    video_names, video_lengths = [], []
    
    print(f"Running evaluation on {len(test_dataset)} videos using weights: {args.weight_path}")

    with torch.no_grad():
        for data in test_dataloader:
            video_fe, text_pfe, vl, e_labels, flow_mag, video_name = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device).squeeze(0), data[4].to(device), data[5]
            video_fe = video_fe.transpose(2, 1)

            predictions, _, _ = model.forward(video_fe, text_pfe, flow_mag)
            scores = torch.sigmoid(predictions.squeeze())
            preds = torch.round(scores)

            test_scores.extend(scores.flatten().tolist())
            test_preds.extend(preds.flatten().tolist())
            test_labels.extend(e_labels.flatten().tolist())
            
            video_names.append(video_name[0])
            video_lengths.append(int(vl.data[0]))

    # Calculate Metrics
    test_AUC, test_mAP = calculate_metrics(test_labels, test_scores)
    print(f"\nFinal Results -> AUC: {test_AUC*100:.2f}% | mAP: {test_mAP*100:.2f}%\n")

    # Save CSVs
    save_dir = os.path.join(os.path.dirname(args.weight_path), "results_csv")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "eval_metrics.txt"), "w") as f:
        f.write(f"AUC: {test_AUC:.4f}\nmAP: {test_mAP:.4f}\n")

    start_idx = 0
    for i in range(len(video_names)):
        v_name = video_names[i].split(".")[0]
        length = video_lengths[i]
        
        with open(os.path.join(save_dir, f"{v_name}_score.csv"), "w") as f:
            csv.writer(f).writerows([[s] for s in test_scores[start_idx : start_idx+length]])
            
        with open(os.path.join(save_dir, f"{v_name}_label.csv"), "w") as f:
            csv.writer(f).writerows([[l] for l in test_labels[start_idx : start_idx+length]])
            
        start_idx += length
        
    print(f"Prediction CSVs saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--weight_path", required=True, type=str, help="Path to best_model_weight.pth")
    parser.add_argument("-exp", default="KIDA_EVAL", type=str)
    parser.add_argument("-dp", "--data_path", default="/memory/luoyuxuan4090/SEDMamba/data", type=str)
    parser.add_argument("-gpu_id", default="cuda:0", type=str)
    parser.add_argument("-w", "--work", default=4, type=int)
    parser.add_argument("-et","--emb_type", default='dinoemb', type=str)
    parser.add_argument("-ep","--emb_path", default='DINOv2', type=str)
    parser.add_argument("-cls", "--num_class", default=1, type=int)
    parser.add_argument("-fd", "--text_dim", default=768, type=int)
    parser.add_argument("-id", "--feature_dim", default=1000, type=int)
    parser.add_argument("-nb", "--num_block", default=3, type=int)
    parser.add_argument("-g", "--com_factor", default=64, type=int)
    main(parser.parse_args())