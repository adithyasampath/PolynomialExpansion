from transformer import Transformer
from dataloader import PolynomialLanguage, PolyDataset
from utils import *

import re
import random
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import evaluate

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.writer = SummaryWriter(f'./logs/transformer_{self.args.exp_name}')
        self.args.dirpath = os.path.join(self.args.dirpath, self.args.exp_name)

        if not os.path.isdir(self.args.dirpath):
            os.makedirs(self.args.dirpath)

        self.create_dataset()
        self.create_dataloaders()

        self.model = Transformer(self.src_lang, self.trg_lang).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_lang.PAD_idx)

        self.bleu = evaluate.load("bleu")
        self.google_bleu = evaluate.load("google_bleu")
        self.rouge = evaluate.load('rouge')
    
    def save_pickle(self, split, src_lang, trg_lang):
        save_to_pickle = {
        f"{split}_src_lang.pickle": src_lang,
        f"{split}_trg_lang.pickle": trg_lang,
        }
        for k, v in save_to_pickle.items():
            with open(os.path.join(self.args.dirpath, k), "wb") as fo:
                pickle.dump(v, fo)

    def create_dataset(self):
        self.dataset_pairs = PolynomialLanguage.load_pairs(self.args.dataset_path)
        self.src_lang, self.trg_lang = PolynomialLanguage.create_vocabs(self.dataset_pairs)
        self.save_pickle("dataset", self.src_lang, self.trg_lang)
        
        self.train_set_pairs = PolynomialLanguage.load_pairs(self.args.train_path)
        self.train_tensors = pairs_to_tensors(self.train_set_pairs, self.src_lang, self.trg_lang)
        self.val_set_pairs = PolynomialLanguage.load_pairs(self.args.val_path)
        self.val_tensors = pairs_to_tensors(self.val_set_pairs, self.src_lang, self.trg_lang)
        self.test_set_pairs = PolynomialLanguage.load_pairs(self.args.test_path)
        self.test_tensors = pairs_to_tensors(self.test_set_pairs, self.src_lang, self.trg_lang)
    
    def create_dataloaders(self):
        self.collate_fn = Collater(self.src_lang, self.trg_lang)
        self.train_dataloader = DataLoader(
            PolyDataset(self.train_tensors),
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
        )
        self.val_dataloader = DataLoader(
            PolyDataset(self.val_tensors),
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
        )
        self.test_dataloader = DataLoader(
            PolyDataset(self.test_tensors),
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
        )


    def train_step(self):
        self.model.train()
        
        epoch_loss = 0
        
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            src, trg = batch
            src, trg = src.to(self.device), trg.to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(src, trg[:,:-1])

            output_dim = output.shape[-1]  
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
                
            loss = self.criterion(output, trg)
            loss.backward()  
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_dataloader)

    def validate_step(self):    
        self.model.eval()
        
        val_epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                src, trg = batch
                src, trg = src.to(self.device), trg.to(self.device)

                output, _ = self.model(src, trg[:,:-1])
                           
                output_dim = output.shape[-1]
                
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                

                loss = self.criterion(output, trg)

                val_epoch_loss += loss.item()
            
        return val_epoch_loss / len(self.val_dataloader)

    def test_step(self):
        # self.load_model()    
        self.model.eval()
        
        test_epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(tqdm(self.test_dataloader)):
                src, trg = batch
                src, trg = src.to(self.device), trg.to(self.device)

                output, _ = self.model(src, trg[:,:-1])
                           
                output_dim = output.shape[-1]
                
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                

                loss = self.criterion(output, trg)

                test_epoch_loss += loss.item()
            
        return test_epoch_loss / len(self.test_dataloader)
    
    def train(self):
        best_val_loss = float('inf')
        for epoch in tqdm(range(self.args.epochs)):
            train_loss = self.train_step()
            print("Epoch: {}, Train Loss: {}".format(epoch, train_loss))   
            self.writer.add_scalar('Train loss (epoch)', train_loss, epoch)

            if epoch % self.args.val_every==0:
                val_loss = self.validate_step()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.args.best_epoch = best_epoch
                    torch.save(self.model.state_dict(), os.path.join(self.args.dirpath, f"best_model_epoch{best_epoch}.pth"))
                    torch.save(self.model, os.path.join(self.args.dirpath, f"best_model_full_epoch{best_epoch}.pth"))
                    print("Epoch: {}, Val Loss: {}".format(epoch, val_loss))
                    self.writer.add_scalar('Val loss (epoch)', val_loss, epoch)   

            # model_file = f"transformer_epoch{epoch}.pth"
            # torch.save(self.model.state_dict(), os.path.join(self.args.dirpath, model_file))
        
        val_loss = self.validate_step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            self.args.best_epoch = best_epoch
            torch.save(self.model.state_dict(), os.path.join(self.args.dirpath, f"best_model_epoch{best_epoch}.pth"))
            torch.save(self.model, os.path.join(self.args.dirpath, f"best_model_full_epoch{best_epoch}.pth"))
        print("Training Complete! Epoch: {}, Val Loss: {}".format(epoch, val_loss))
        self.writer.add_scalar('Val loss (epoch)', val_loss, epoch)   


    def predict(self, sentences, batch_size=128):
        """Efficiently predict a list of sentences"""
        pred_tensors = [
            sentence_to_tensor(sentence, self.src_lang)
            for sentence in tqdm(sentences, desc="creating prediction tensors")
        ]

        collate_fn = Collater(self.src_lang, predict=True)
        pred_dataloader = DataLoader(
            PolyDataset(pred_tensors),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        self.model.eval()
        
        with torch.no_grad():
            sentences = []
            words = []
            attention = []

            for batch in tqdm(pred_dataloader, desc="predict batch num"):
                preds = self.predict_batch(batch.to(self.device))
                pred_sentences, pred_words, pred_attention = preds
                sentences.extend(pred_sentences)
                words.extend(pred_words)
                attention.extend(pred_attention)

        # sentences = [num pred sentences]
        # words = [num pred sentences, trg len]
        # attention = [num pred sentences, n heads, trg len, src len]

        return sentences, words, attention

    def predict_batch(self, batch):
        """Predicts on a batch of src_tensors."""
        # batch = src_tensor when predicting = [batch_size, src len]

        src_tensor = batch
        src_mask = self.model.make_src_mask(batch)

        # src_mask = [batch size, 1, 1, src len]

        enc_src = self.model.encoder(src_tensor, src_mask)

        # enc_src = [batch size, src len, hid dim]

        trg_indexes = [[self.trg_lang.SOS_idx] for _ in range(len(batch))]

        # trg_indexes = [batch_size, cur trg len = 1]

        trg_tensor = torch.LongTensor(trg_indexes).to(self.device)

        # trg_tensor = [batch_size, cur trg len = 1]
        # cur trg len increases during the for loop up to the max len

        for _ in range(self.args.max_len):

            trg_mask = self.model.make_trg_mask(trg_tensor)

            # trg_mask = [batch size, 1, cur trg len, cur trg len]

            output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output = [batch size, cur trg len, output dim]

            preds = output.argmax(2)[:, -1].reshape(-1, 1)

            # preds = [batch_size, 1]

            trg_tensor = torch.cat((trg_tensor, preds), dim=-1)

            # trg_tensor = [batch_size, cur trg len], cur trg len increased by 1

        src_tensor = src_tensor.detach().cpu().numpy()
        trg_tensor = trg_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_words = []
        pred_sentences = []
        pred_attention = []
        for src_indexes, trg_indexes, attn in zip(src_tensor, trg_tensor, attention):
            # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
            # src_indexes = [src len = len of longest sentence (padded if not longest)]

            # indexes where first eos tokens appear
            src_eosi = np.where(src_indexes == self.src_lang.EOS_idx)[0][0]
            _trg_eosi_arr = np.where(trg_indexes == self.trg_lang.EOS_idx)[0]
            if len(_trg_eosi_arr) > 0:  # check that an eos token exists in trg
                trg_eosi = _trg_eosi_arr[0]
            else:
                trg_eosi = len(trg_indexes)

            # cut target indexes up to first eos token and also exclude sos token
            trg_indexes = trg_indexes[1:trg_eosi]

            # attn = [n heads, trg len=max len, src len=max len of sentence in batch]
            # we want to keep n heads, but we'll cut trg len and src len up to
            # their first eos token
            attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

            words = [self.trg_lang.index2word[index] for index in trg_indexes]
            sentence = self.trg_lang.words_to_sentence(words)
            pred_words.append(words)
            pred_sentences.append(sentence)
            pred_attention.append(attn)

        # pred_sentences = [batch_size]
        # pred_words = [batch_size, trg len]
        # attention = [batch size, n heads, trg len (varies), src len (varies)]

        return pred_sentences, pred_words, pred_attention

    def load_model(self):
        self.args.model_path = os.path.join(self.args.dirpath, f"best_model_full_epoch{self.args.best_epoch}.pth")
        self.model = torch.load(self.args.model_path)
        # self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()
        print(f'Loaded model {self.args.exp_name} from: {self.args.model_path}\nThe model has {self.count_parameters():,} trainable parameters')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def test_predict(self, split="test"):
        self.model.eval()
        if split == "test":
            _dataloader = self.test_dataloader
        elif split == "val":
            _dataloader = self.val_dataloader

        with torch.no_grad():

            sentences = []
            words = []
            attention = []

            for i, batch in enumerate(tqdm(_dataloader)):
                src, trg = batch

                preds = self.predict_batch(src.to(self.device))
                pred_sentences, pred_words, pred_attention = preds

                sentences.extend(pred_sentences)
                words.extend(pred_words)
                attention.extend(pred_attention)
            
        return sentences, words, attention

    def test_metrics(self, split="test"):
        self.load_model()  

        if split == "test":
            src_sentences, trg_sentences = zip(*self.test_set_pairs)
        elif split == "val":
            src_sentences, trg_sentences = zip(*self.val_set_pairs)

        # prd_sentences, _, _ = self.predict(src_sentences, batch_size=self.args.batch_size)
        prd_sentences, _, _ = self.test_predict(split=split)
        assert len(prd_sentences) == len(src_sentences) == len(trg_sentences)

        total_score = 0
        for i, (src, trg, prd) in enumerate(
            tqdm(
                zip(src_sentences, trg_sentences, prd_sentences),
                desc="scoring",
                total=len(src_sentences),
            )
        ):
            pred_score = score(trg, prd)
            total_score += pred_score
            if i < 10:
                print(f"\n\n\n---- Example {i} ----")
                print(f"src = {src}")
                print(f"trg = {trg}")
                print(f"prd = {prd}")
                print(f"score = {pred_score}")

        final_score = total_score / len(prd_sentences)
        print(f"{total_score}/{len(prd_sentences)} = {final_score:.4f}")
        # bleu_score = self.bleu.compute(predictions=prd_sentences, references=src_sentences)
        # google_bleu_score = self.bleu.compute(predictions=prd_sentences, references=src_sentences)
        # rouge_score = self.rouge.compute(predictions=prd_sentences, references=src_sentences)
        # final_result = {"accuracy": final_score, "bleu": bleu_score, "google_bleu": google_bleu_score, "rouge": rouge_score}
        
        # with open(f'{self.args.exp_name}.json', 'w') as f:
        #     json.dump(final_result, f)
        # return final_result
        return final_score

    