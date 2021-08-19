import collections
import json
import os
import glob
import nltk
import torch
import numpy as np
import faiss
import requests
import io

from CLIP import clip
from nltk.tokenize import wordpunct_tokenize
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from argparse import ArgumentParser
from autofaiss.external.quantize import Quantizer

import retrofit


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder : str,
                 args):
        super().__init__()
        self.prefix = args.prefix
        self.lower = args.lower
        self.ext = args.ext

        path = Path(folder)
        text_files = sorted([*path.glob(f'**/*.{self.ext}')])

        data = []
        tokens = []
        for f in text_files:
            try:
                text = f.read_text().split('\n')
            except UnicodeDecodeError:
                continue
            text = list(filter(lambda t: len(t) > 0, text))
            if args.use_unigrams or args.use_bigrams or args.use_trigrams:
                for line in text:
                    if args.lower:
                        tokens.extend([w.lower() for w in wordpunct_tokenize(line)])
                    else:
                        tokens.extend([w for w in wordpunct_tokenize(line)])
            if args.use_line:
                if args.lower:
                    text = [t.lower() for t in text]
                data.extend(text)
    
        if args.use_unigrams:
            unigrams = collections.Counter(tokens).most_common(args.topk_ngrams)
            unigrams = [t[0] for t in unigrams]
            data.extend(unigrams)
     
        if args.use_bigrams:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            bigrams = finder.nbest(bigram_measures.pmi, args.topk_ngrams)
            bigrams = [' '.join(g) for g in bigrams]
            data.extend(bigrams)

        if args.use_trigrams:
            trigram_measures = nltk.collocations.TrigramAssocMeasures()
            finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            trigrams = finder.nbest(trigram_measures.pmi, args.topk_ngrams)
            trigrams = [' '.join(g) for g in trigrams]
            data.extend(trigrams)
        
        print(f'Number of entries: {len(data)}')
        self.data = data

    def __getitem__(self, idx):
        item = self.prefix + self.data[idx]
        return item, self.data[idx]
    
    def __len__(self):
        return len(self.data)


def load_index(args):
    index = faiss.read_index(glob.glob(f"{args.index_dir}/*.index")[0])
    return index


def encode(args, net):
    text_embeddings = []
    entries = []
    dataset = TextDataset(folder=args.text_dir, args=args)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_prepro_workers,
                      pin_memory=True,
                      prefetch_factor=2)
    print('Encoding with Conditional CLIP...')
    batches_seen = 0
    chunks_count = 0
    batches_per_chunk = args.chunk_size // args.batch_size
    for item, entry in tqdm(data):
        text_features = net.encode_text(item)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_features.cpu().numpy().astype('float32'))
        batches_seen += 1
        if batches_seen % batches_per_chunk == 0:
            emb = np.concatenate(text_embeddings)
            idx = str(chunks_count).zfill(5)
            fname = os.path.join(args.index_dir, f'emb{idx}.npy')
            np.save(fname, emb)
            chunks_count += 1
            text_embeddings = []
        if args.save_entries:
            entries.extend(entry)
    emb = np.concatenate(text_embeddings)
    idx = str(chunks_count).zfill(5)
    fname = os.path.join(args.index_dir, f'emb{idx}.npy')
    np.save(fname, emb)

    # Store entries if applicable
    if args.save_entries:
        print('Saving entries...')
        fname = os.path.join(args.index_dir, 'entries.txt')
        with open(fname, 'w') as f:
            for line in entries:
                f.write(f'{line}\n')


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--text_dir', type=str, default=None)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--ext', type=str, default='tgt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chunk_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save_entries', type=bool, default=False)
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--use_line', type=bool, default=False)
    parser.add_argument('--use_unigrams', type=bool, default=False)
    parser.add_argument('--use_bigrams', type=bool, default=False)
    parser.add_argument('--use_trigrams', type=bool, default=False)
    parser.add_argument('--topk_ngrams', type=int, default=10000)
    parser.add_argument('--filter', type=int, default=3)
    parser.add_argument('--metric_type', type=str, default='ip')
    parser.add_argument('--max_index_memory_usage', type=str, default='32GB')
    parser.add_argument('--current_memory_available', type=str, default='32GB')
    parser.add_argument('--max_index_query_time_ms', type=int, default=10)
    args = parser.parse_args()

    # Load network and encode
    config = dotdict(torch.load(args.config))
    net = retrofit.load_params(config).to(args.device)
    encode(args, net)

    # Store args
    fname = os.path.join(args.index_dir, 'args.txt')
    with open(fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Compute index
    quantizer = Quantizer()
    quantizer.quantize(embeddings_path=args.index_dir,
                       output_path=args.index_dir,
                       metric_type=args.metric_type,
                       max_index_memory_usage=args.max_index_memory_usage,
                       current_memory_available=args.current_memory_available,
                       max_index_query_time_ms=args.max_index_query_time_ms)


if __name__ == '__main__':
    main()