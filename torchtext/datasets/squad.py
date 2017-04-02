import os
import json
import linecache
import nltk
import numpy as np
import os
import sys
from tqdm import tqdm
import random
from collections import Counter
from six.moves.urllib.request import urlretrieve
from spacy.en import English

from .. import data

base_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
train_filename = "train-v1.1.json"
train_size = 30288272
dev_filename = "dev-v1.1.json"
dev_size = 4854279

def reporthook(t):
  """https://github.com/tqdm/tqdm"""
  last_b = [0]

  def inner(b=1, bsize=1, tsize=None):
    """
    b: int, optionala
        Number of blocks just transferred [default: 1].
    bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner

def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix,filename), reporthook=reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    else:
       local_filename = os.path.join(prefix, filename)
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix,filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename

def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def list_topics(data):
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    return list_topics


def tokenizer():
    nlp = English()

    def tokenize(sequence):
        doc = nlp(sequence)
        tokens = [token.text.replace("``", '"').replace("''", '"') for token in doc]
        return tokens

    return tokenize


def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):
        if char != u' ':
            acc += char
            context_token = context_tokens[current_token_idx]
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map


def invert_map(answer_map):
    return {v[1]: [v[0], k] for k, v in answer_map.iteritems()}


def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0
    tokenize = tokenizer()

    with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(prefix, tier +'.question'), 'w') as question_file,\
         open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
         open(os.path.join(prefix, tier +'.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)

                    answers = qas[qid]['answers']
                    qn += 1

                    num_answers = range(1)

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']
                        a_s = qas[qid]['answers'][ans_id]['answer_start']

                        text_tokens = tokenize(text)

                        answer_start = qas[qid]['answers'][ans_id]['answer_start']

                        answer_end = answer_start + len(text)

                        last_word_answer = len(text_tokens[-1]) # add one to get the first char

                        try:
                            a_start_idx = answer_map[answer_start][1]

                            a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            text_file.write(' '.join(text_tokens) + '\n')
                            span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')

                        except Exception as e:
                            skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an


def save_files(prefix, tier, indices):
  with open(os.path.join(prefix, tier + '.context'), 'w') as context_file,  \
     open(os.path.join(prefix, tier + '.question'), 'w') as question_file,\
     open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
     open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

    for i in indices:
      context_file.write(linecache.getline(os.path.join(prefix, 'train.context'), i))
      question_file.write(linecache.getline(os.path.join(prefix, 'train.question'), i))
      text_file.write(linecache.getline(os.path.join(prefix, 'train.answer'), i))
      span_file.write(linecache.getline(os.path.join(prefix, 'train.span'), i))


def split_tier(prefix, train_percentage = 0.9, shuffle=False):
    # Get number of lines in file
    context_filename = os.path.join(prefix, 'train' + '.context')
    # Get the number of lines
    with open(context_filename) as current_file:
        num_lines = sum(1 for line in current_file)
    # Get indices and split into two files
    indices_dev = list(range(num_lines)[int(num_lines * train_percentage)::])
    if shuffle:
        np.random.shuffle(indices_dev)
        print("Shuffling...")
    save_files(prefix, 'val', indices_dev)
    indices_train = list(range(num_lines)[:int(num_lines * train_percentage)])
    if shuffle:
        np.random.shuffle(indices_train)
    save_files(prefix, 'train', indices_train)


def maybe_download_all(root):
    if not os.path.exists(root):
        os.makedirs(root)
    train = maybe_download(base_url, train_filename, root, num_bytes=train_size)
    dev = maybe_download(base_url, dev_filename, root, num_bytes=dev_size)
    return train, dev

def maybe_preprocess(root, dl_train, dl_dev):

    e = os.path.exists
    j = os.path.join

    if not e(root):
        os.makedirs(root)

    vc, vq, va, vs = j(root, "val.context"), j(root, "val.question"),\
                     j(root, "val.answer"), j(root, "val.span")
    tc, tq, ta, ts = j(root, "train.context"), j(root, "train.question"),\
                     j(root, "train.answer"), j(root, "train.span")

    if e(vc) and e(vq) and e(va) and e(vs) and e(tc) and e(tq) and e(ta) and e(ts):
        return

    train_data = data_from_json(dl_train)
    train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', root)
    print("Processed {} questions and {} answers in train".format(train_num_questions, train_num_answers))

    split_tier(root, 0.95, shuffle=True)
    print("Split the dataset into train and validation")

    dev_data = data_from_json(dl_dev)
    dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', root)
    print("Processed {} questions and {} answers in dev".format(dev_num_questions, dev_num_answers))
    

def verify(root):
    download_prefix = os.path.join(root, "download", "squad")
    data_prefix = os.path.join(root, "data", "squad")
    dl_train, dl_dev = maybe_download_all(download_prefix)

    maybe_preprocess(data_prefix, dl_train, dl_dev)
    


class SQUAD(data.Dataset):
    """Loads the SQuAD QA dataset."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, fields=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        verify(path)
        prefix = os.path.join(path, "data", "squad", "train")
        fields = ["context", "question", "answer", "span"]
        train_filenames = [prefix + '.' + field for field in fields]
        train_files = [open(f) for f in train_filenames]

        c, q, a, s = tuple(train_files)

        examples = []
        for context, query, answer, span in zip(c, q, a, s):
            context, query, answer, span = \
                context.strip(), query.strip(), answer.strip(), span.strip()
            if context != '' and query != '':
                examples.append([context, query, answer, span])


        self.examples = examples
        self.fields = {}
