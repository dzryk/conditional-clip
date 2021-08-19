import json

from collections import defaultdict
from argparse import ArgumentParser


def save_text(datadir, split):
    """Loads the questions/answers jsons, extract and store."""

    # Questions and answers
    qpath = f'{datadir}/v2_OpenEnded_mscoco_{split}_questions.json'
    apath = f'{datadir}/v2_mscoco_{split}_annotations.json'
    prefix = f'COCO_{split}'
    with open(qpath, 'r') as f:
        questions = json.load(f)
    with open(apath, 'r') as f:
        answers = json.load(f)
    for (x, y) in zip(questions['questions'], answers['annotations']):
        key = x['image_id']
        qvalue = x['question']
        avalue = y['multiple_choice_answer']
        fname = f'{prefix}_{str(key).zfill(12)}.txt'
        path = f'{datadir}/{split}/{fname}'
        with open(path, 'a') as f:
            f.write(f'{qvalue}\t{avalue}\n')


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()

    save_text(args.datadir, 'train2014')
    save_text(args.datadir, 'val2014')


if __name__ == '__main__':
    main()