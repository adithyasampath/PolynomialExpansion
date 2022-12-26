import argparse
from trainer import Trainer
from utils import create_dataset
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=29)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=3)
    parser.add_argument("--enc_heads", type=int, default=8)
    parser.add_argument("--dec_heads", type=int, default=8)
    parser.add_argument("--enc_pf_dim", type=int, default=256)
    parser.add_argument("--dec_pf_dim", type=int, default=256)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dirpath", type=str, default="models/transformer")
    parser.add_argument("--dataset_path", type=str, default="dataset/dataset.txt")
    parser.add_argument("--train_path", type=str, default="dataset/train.txt")
    parser.add_argument("--val_path", type=str, default="dataset/val.txt")
    parser.add_argument("--test_path", type=str, default="dataset/test.txt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=51)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument("--model_path", type=str, default="models/transformer/nlayers3hdim256/best_model_full_epoch50.pth")
    parser.add_argument("--best_epoch", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="nlayers3hdim256")
    args = parser.parse_args()

    poly_trainer = Trainer(args)

    # create_dataset(root_folder="./dataset", seed=args.seed)

    if not args.test:
        poly_trainer.train()

    test_metrics = poly_trainer.test_metrics(split="test")
    val_metrics = poly_trainer.test_metrics(split="val")
    print(f"Test accuracy: {test_metrics}\tVal accuracy: {val_metrics}")
    
    final_result = {"test_accuracy": test_metrics, "val_accuracy": val_metrics}
        
    with open(f'{args.exp_name}.json', 'w') as f:
        json.dump(final_result, f)