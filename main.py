import torch
import torch.nn as nn
import argparse
import random
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, PreTrainedModel
from collections import Counter

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--prob', type=float, default=0.2)
parser.add_argument('--max_mask', type=int, default=3)
parser.add_argument('--min_len', type=int, default=3)

parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--label_smooth', type=float, default=0.)

parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'])
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_enc_layers', type=int, default=6)
parser.add_argument('--num_dec_layers', type=int, default=6)
parser.add_argument('--dim_ff', type=int, default=1024)

parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default='models')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--is_train', type=bool, default=False)
parser.add_argument('--is_eval', type=bool, default=False)

args = parser.parse_args()
print(args)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def generate_square_subsequent_mask(sz, device='cuda'):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class CustomTokenizer:
    def __init__(self):
        self.vocab = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.token2idx_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}

    def build_vocab(self, data):
        for row in data:
            for word in row:
                if word in self.token2idx_dict:
                    continue
                self.token2idx_dict[word] = len(self.token2idx_dict)
        tmp_vocab = {v: k for k, v in self.token2idx_dict.items()}
        self.vocab.update(tmp_vocab)

    def token2idx(self, tok):
        if tok not in self.token2idx_dict:
            tok = '[UNK]'
        return self.token2idx_dict[tok]

    def idx2token(self, idx):
        if idx > len(self.vocab):
            return '[UNK]'
        else:
            return self.vocab[idx]

    def encode(self, tokens):
        res = [self.token2idx_dict['[CLS]']]
        for tok in tokens:
            res.append(self.token2idx(tok))
        res.append(self.token2idx_dict['[SEP]'])
        return res

    def decode(self, indices, without_special=False):
        res = []
        for idx in indices:
            if without_special and idx in [0, 2, 3]:
                continue
            res.append(self.idx2token(int(idx)))
        return res


class CustomSeq2SeqModel(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, out_tokenizer, num_layers=2, embed_dim=256, hidden_dim=256,
                 max_len=128, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.tokenizer = out_tokenizer
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_layers = num_layers

        self.input_embedding = nn.Embedding(num_embeddings=in_vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.output_embedding = nn.Embedding(num_embeddings=out_vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                                     dropout=dropout, bidirectional=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim * 2, num_layers=num_layers, batch_first=True,
                                     dropout=dropout, bidirectional=False)
        self.linear = nn.Linear(hidden_dim * 2, out_vocab_size)

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        in_embed = self.input_embedding(inputs.to(self.device))
        _, (h, c) = self.encoder(in_embed)
        # packed_in_embed = nn.utils.rnn.pack_padded_sequence(in_embed, torch.LongTensor([in_embed.size(1)]).tile(batch_size), batch_first=True)
        # enc_out, (h, c) = self.encoder(packed_in_embed)
        # enc_out_unpack, enc_lens_unpack = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)

        decoder_inputs = nn.functional.one_hot(torch.LongTensor([self.tokenizer.token2idx('[CLS]')] * batch_size),
                                                     num_classes=self.out_vocab_size).unsqueeze(1).to(self.device)

        h = h.reshape(self.num_layers, 2, batch_size, self.hidden_dim).permute(0, 2, 1, 3)\
            .reshape(self.num_layers,batch_size, self.hidden_dim * 2)
        c = c.reshape(self.num_layers, 2, batch_size, self.hidden_dim).permute(0, 2, 1, 3)\
            .reshape(self.num_layers, batch_size, self.hidden_dim * 2)

        for t in range(1, self.max_len):
            out_embed = self.output_embedding(decoder_inputs[:, -1:].argmax(dim=-1))
            dec_out, (h, c) = self.decoder(out_embed, (h, c))
            pred = self.linear(dec_out)
            decoder_inputs = torch.cat([decoder_inputs, pred], dim=1)

        return decoder_inputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=args.max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class CustomTransformerModel(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, in_tokenizer, out_tokenizer, d_model=256, nhead=8, num_enc_layers=6, num_dec_layers=6,
                 dim_ff=1024, dropout=0.1, device='cuda'):
        super(CustomTransformerModel, self).__init__()
        self.d_model = d_model
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.device = device

        self.in_embedding = nn.Embedding(num_embeddings=in_vocab_size, embedding_dim=d_model)
        self.out_embedding = nn.Embedding(num_embeddings=out_vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=args.max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers, dim_feedforward=dim_ff, dropout=dropout)
        self.linear = nn.Linear(d_model, out_vocab_size)

    def forward(self, inputs, targets):
        batch_size, seq_len = inputs.shape
        in_pad_idx = self.in_tokenizer.token2idx('[PAD]')
        out_pad_idx = self.out_tokenizer.token2idx('[PAD]')

        input_masks = torch.zeros((inputs.shape[1], inputs.shape[1])).to(self.device).type(torch.bool)
        target_masks = generate_square_subsequent_mask(targets.shape[1])
        input_pad_masks = (inputs == in_pad_idx)
        target_pad_masks = (targets == out_pad_idx)

        in_embed = self.in_embedding(inputs.transpose(0, 1).to(self.device)) * math.sqrt(self.d_model)
        in_embed = self.positional_encoding(in_embed)
        out_embed = self.out_embedding(targets.transpose(0, 1).to(self.device)) * math.sqrt(self.d_model)
        out_embed = self.positional_encoding(out_embed)

        trans_out = self.transformer(in_embed, out_embed,
                                     src_mask=input_masks, tgt_mask=target_masks,
                                     src_key_padding_mask=input_pad_masks, tgt_key_padding_mask=target_pad_masks)
        outputs = self.linear(trans_out)        # T, N, E

        return outputs.transpose(0, 1)


# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=-100):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim
#         self.ignore_index = ignore_index
#
#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def compute_loss(preds, targets, device='cuda'):
    batch_size = preds.shape[0]
    cse = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smooth)
    targets = torch.cat([targets.long(), torch.zeros(batch_size, preds.shape[1]-targets.shape[1]).long().to(device)], dim=1)
    loss = cse(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1).to(device))

    return loss


def exact_match(preds, targets, tokenizer):
    cnt = 0
    not_matched = []
    for p, t in zip(preds, targets):
        dp = tokenizer.decode(p.argmax(dim=-1), without_special=True)
        dt = tokenizer.decode(t, without_special=True)
        if dp == dt:
            cnt += 1
        else:
            not_matched.append({'Predict': dp, 'Target ': dt})

    return cnt / len(preds), not_matched


def sentence_similarity(preds, targets, tokenizer):
    word_cnt = 0
    len_dp = 0
    len_dt = 0
    for p, t in zip(preds, targets):
        dp = tokenizer.decode(p.argmax(dim=-1), without_special=True)
        dt = tokenizer.decode(t, without_special=True)
        len_dp += len(dp)
        len_dt += len(dt)

        for i in range(min(len(dp), len(dt))):
            if dp[i] == dt[i]:
                word_cnt += 1

    prec = (word_cnt / len_dp) if len_dp > 0 else 0
    recall = (word_cnt / len_dt) if len_dt > 0 else 0

    return prec, recall


def intersection_over_union(preds, targets, tokenizer):
    ious = []
    for p, t in zip(preds, targets):
        dp = tokenizer.decode(p.argmax(dim=-1), without_special=True)
        dt = tokenizer.decode(t, without_special=True)
        dp_counter = Counter(dp)
        dt_counter = Counter(dt)

        union = {}
        intersection = {}
        tokens = set(dp + dt)

        for tok in tokens:
            if tok not in union:
                union[tok] = 0
            union[tok] = max(dp_counter[tok], union[tok]) if tok in dp_counter else union[tok]
            union[tok] = max(dt_counter[tok], union[tok]) if tok in dt_counter else union[tok]

            if (tok in dp_counter) and (tok in dt_counter):
                intersection[tok] = min(dp_counter[tok], dt_counter[tok])

        ious.append((sum(intersection.values()) / sum(union.values())) if sum(union.values()) != 0 else 0)

    return sum(ious) / len(ious)


def metrics(preds, targets, tokenizer):
    preds = preds.cpu().detach()
    targets = targets.cpu().detach()

    em_score, not_matched = exact_match(preds, targets, tokenizer)
    sim_prec, sim_recall = sentence_similarity(preds, targets, tokenizer)
    iou_score = intersection_over_union(preds, targets, tokenizer)

    return em_score, sim_prec, sim_recall, iou_score, not_matched


def evaluating(model, test_data, test_target, tokenizer, dataset_type='test', batch_size=64, device='cuda', show_examples=False):
    test_loss = 0.
    em_score = 0
    sim_prec = 0
    sim_recall = 0
    iou_score = 0
    with torch.no_grad():
        for i in tqdm(range((len(test_data) + batch_size - 1) // batch_size)):
            test_inputs = test_data[i * batch_size:min(i * batch_size + batch_size, len(test_data))].to(args.device)
            test_targets = test_target[i * batch_size:min(i * batch_size + batch_size, len(test_target))].to(args.device)

            if isinstance(model, CustomTransformerModel):
                target_input = test_targets[:, :-1]
                test_targets_y = test_targets[:, 1:]
                test_preds = model(test_inputs, target_input)
            else:
                test_preds = model(test_inputs)
                test_targets_y = test_targets
            test_loss += compute_loss(test_preds, test_targets_y, device=device).item()

            em, sp, sr, iou, not_matched = metrics(test_preds, test_targets_y, tokenizer)
            em_score += em * test_inputs.shape[0]
            sim_prec += sp * test_inputs.shape[0]
            sim_recall += sr * test_inputs.shape[0]
            iou_score += iou * test_inputs.shape[0]

    print(f'{dataset_type}_loss: {test_loss / len(test_data)}')
    print(f'{dataset_type}_exact_match: {em_score / len(test_data)}')
    print(f'{dataset_type}_similarity_prec: {sim_prec / len(test_data)}')
    print(f'{dataset_type}_similarity_recall: {sim_recall / len(test_data)}')
    print(f'{dataset_type}_iou: {iou_score / len(test_data)}')
    if show_examples:
        for x in not_matched[:10]:
            for k, v in x.items():
                print(f'{k}: {v}')
            print()


def training(model, train_data, train_target, test_data, test_target, tokenizer, batch_size=64, epoch=10, lr=1e-3,
             eps=1e-5, momentum=0.9, weight_decay=1e-5, save_path='models', is_eval=False, optim='adamw', device='cuda'):
    total_loss = 0.
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    prev_em_score = 0.
    for e in range(epoch):
        # TRAIN
        em_score = 0
        sim_prec = 0
        sim_recall = 0
        iou_score = 0
        # flag = False
        for i in tqdm(range((len(train_data) + batch_size - 1) // batch_size)):
            inputs = train_data[i*batch_size:min(i*batch_size+batch_size, len(train_data))].to(args.device)
            targets = train_target[i*batch_size:min(i*batch_size+batch_size, len(train_target))].to(args.device)

            optimizer.zero_grad()
            if isinstance(model, CustomTransformerModel):
                target_input = targets[:, :-1]
                targets_y = targets[:, 1:]
                preds = model(inputs, target_input)
            else:
                preds = model(inputs)
                targets_y = targets
            loss = compute_loss(preds, targets_y, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            em, sp, sr, iou, not_matched = metrics(preds, targets_y, tokenizer)
            em_score += em * inputs.shape[0]
            sim_prec += sp * inputs.shape[0]
            sim_recall += sr * inputs.shape[0]
            iou_score += iou * inputs.shape[0]

        print(f'epoch: {e}')
        print(f'train_loss: {total_loss / len(train_data)}')
        print(f'train_exact_match: {em_score / len(train_data)}')
        print(f'train_similarity_prec: {sim_prec / len(train_data)}')
        print(f'train_similarity_recall: {sim_recall / len(train_data)}')
        print(f'train_iou: {iou_score / len(train_data)}')
        total_loss = 0.

        # VALIDATION
        if is_eval:
            evaluating(model, test_data, test_target, tokenizer, batch_size=batch_size, device=device)

        # SAVE MODEL
        if e % 10 == 0:
            torch.save(model.state_dict(), save_path + f'/{e}')
        if em_score > prev_em_score:
            torch.save(model.state_dict(), save_path + f'/best_{e}')
            prev_em_score = em_score

    return model


def preprocessing(data, prob=0.2, max_mask=3, min_len=3):
    for sen in data:
        n_drop = 0
        if len(sen) > max_mask + min_len:
            while np.random.uniform(0, 1) < prob and n_drop < max_mask:
                idx = np.random.choice(len(sen))
                sen[int(idx)] = '[UNK]'
                n_drop += 1

    return data


def main():
    # GET DATA
    with open('train_source.txt', 'r') as f:
        raw_data = f.readlines()
    with open('train_target.txt', 'r') as f:
        raw_target = f.readlines()
    with open('test_source.txt', 'r') as f:
        raw_test_data = f.readlines()
    with open('test_target.txt', 'r') as f:
        raw_test_target = f.readlines()

    data, target, test_data, test_target = [], [], [], []
    for line in raw_data:
        data.append([x for x in line.split()])
    for line in raw_target:
        target.append([x for x in line.split()])
    for line in raw_test_data:
        test_data.append([x for x in line.split()])
    for line in raw_test_target:
        test_target.append([x for x in line.split()])

    # MANIPULATE DATA
    data = preprocessing(data, prob=args.prob, max_mask=args.max_mask, min_len=args.min_len)

    # TOKENIZER
    in_tokenizer = CustomTokenizer()
    in_tokenizer.build_vocab(data)
    out_tokenizer = CustomTokenizer()
    out_tokenizer.build_vocab(target)

    tokenized_data = nn.utils.rnn.pad_sequence([torch.LongTensor(in_tokenizer.encode(x)) for x in data], batch_first=True, padding_value=0)
    tokenized_target = nn.utils.rnn.pad_sequence([torch.LongTensor(out_tokenizer.encode(x)) for x in target], batch_first=True, padding_value=0)
    tokenized_test_data = nn.utils.rnn.pad_sequence([torch.LongTensor(in_tokenizer.encode(x)) for x in test_data], batch_first=True, padding_value=0)
    tokenized_test_target = nn.utils.rnn.pad_sequence([torch.LongTensor(out_tokenizer.encode(x)) for x in test_target], batch_first=True, padding_value=0)


    # MODEL - biLSTM to LSTM
    if args.model_type == 'lstm':
        model = CustomSeq2SeqModel(len(in_tokenizer.vocab), len(out_tokenizer.vocab), out_tokenizer,
                               num_layers=args.num_layers, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                               max_len=args.max_len, dropout=args.dropout, device=args.device).to(args.device)

    # MODEL - Transformer
    elif args.model_type == 'transformer':
        model = CustomTransformerModel(len(in_tokenizer.vocab), len(out_tokenizer.vocab),
                                       in_tokenizer, out_tokenizer,
                                       d_model=args.d_model, nhead=args.nhead, num_enc_layers=args.num_enc_layers,
                                       num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff, dropout=args.dropout,
                                       device=args.device).to(args.device)
        # for p in model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    # TRAINING
    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path))

    if args.is_train:
        model = training(model, tokenized_data, tokenized_target, tokenized_test_data, tokenized_test_target,
                         out_tokenizer, batch_size=args.batch_size, epoch=args.epoch, lr=args.lr, eps=args.eps,
                         momentum=args.momentum, weight_decay=args.weight_decay, save_path=args.save_path,
                         is_eval=args.is_eval, optim=args.optim, device=args.device)

    if args.is_eval and not args.is_train:
        evaluating(model, tokenized_data, tokenized_target, out_tokenizer, dataset_type='train', batch_size=args.batch_size, device=args.device)
        evaluating(model, tokenized_test_data, tokenized_test_target, out_tokenizer, batch_size=args.batch_size, device=args.device, show_examples=True)


if __name__ == '__main__':
    main()