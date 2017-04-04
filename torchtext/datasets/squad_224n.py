import os
import json
import linecache
import numpy as np
import os
import sys
import _pickle as pickle
from tqdm import tqdm
import random
from collections import Counter
from six.moves.urllib.request import urlretrieve
from spacy.en import English

from .. import data
from .. import utils

base_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
train_filename = "train-v1.1.json"
train_size = 30288272
dev_filename = "dev-v1.1.json"
dev_size = 4854279


def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    if os.path.exists(os.path.join(prefix, filename)):
        local_filename = os.path.join(prefix, filename)
        file_stats = os.stat(local_filename)
        if num_bytes is None or file_stats.st_size == num_bytes:
            return local_filename

    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix, filename),
                                                reporthook=utils.reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e

    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
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
    list_topics = [data['data'][idx]['title'] for idx in range(0, len(data['data']))]
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

    with open(os.path.join(prefix, tier + '.context'), 'w') as context_file, \
         open(os.path.join(prefix, tier + '.question'), 'w') as question_file,\
         open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
         open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

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

                        last_word_answer = len(text_tokens[-1])  # add one to get the first char

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
    return qn, an


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


def split_tier(prefix, train_percentage=0.9, shuffle=False):
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
    dc, dq, da, ds = j(root, "dev.context"), j(root, "dev.question"),\
        j(root, "dev.answer"), j(root, "dev.span")
#    vic, viq, tic, tiq, dic, diq  = j(root, "val.ids.context"), j(root, "val.ids.question"),\
#                                    j(root, "train.ids.context"), j(root, "train.ids.question") \
#                                    j(root, "dev.ids.context"), j(root, "dev.ids.question") \
#  [vic, viq, tic, tiq, dic, diq]
    files = [vc, vq, va, vs, tc, tq, ta, ts, dc, dq, da, ds]
    efiles = [f for f in files if not e(f)]
    if len(efiles) == 0:
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
        return data.interleave_keys(len(ex.context), len(ex.question))

    def __init__(self, path, tier="train", fields=None, **kwargs):
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
        prefix = os.path.join(path, "data", "squad", tier)
        cache_file = prefix + "_examples.pickle"

        fieldnames = ["context", "question", "answer", "span"]
        if fields is None:
            raise Exception('expected 3 fields')
        if len(fields) != 3:
            raise Exception('expected 3 fields')
        if not isinstance(fields[0], (tuple, list)):
            fields = [("context", fields[0]), ("question", fields[0]), ("answer", fields[1]), ("span", fields[2])]

        # loading pickled version saves ~15s (75%) on
        # load time
        print("loading {} examples".format(tier))
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                examples = pickle.load(f)
                super(SQUAD, self).__init__(examples, fields, **kwargs)
                return

        train_filenames = [prefix + '.' + field for field in fieldnames]
        train_files = [open(f) for f in train_filenames]

        c, q, a, s = tuple(train_files)

        examples = []
        for context, query, answer, span in zip(c, q, a, s):
            context, query, answer, span = \
                context.strip(), query.strip(), answer.strip(), span.strip()
            if context != '' and query != '':
                examples.append(data.Example.fromlist(
                    [context, query, answer, span], fields))
        with open(cache_file, "wb") as f:
            pickle.dump(examples, f)

        super(SQUAD, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, context_field, span_field, answer_field, root='.'):
        fields = [context_field, span_field, answer_field]
        return (cls(root, tier="train", fields=fields),
                cls(root, tier="val", fields=fields),
                cls(root, tier="dev", fields=fields))
