import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter
from tqdm import tqdm
import numpy as np
import _pickle as pickle
from torchtext.data import Field
import torchtext.data as textdata
import re

def npz_save(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez(name+".npz", keys=keys, values=values)

def npz_load(filename):
    npzfile = np.load(filename+".npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))

class SQUAD_config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def get_phrase(context, wordss, span):
    """
    Obtain phrase as substring of context given start and stop indices in word level
    :param context:
    :param wordss:
    :param start: [sent_idx, word_idx]
    :param stop: [sent_idx, word_idx]
    :return:
    """
    start, stop = span
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]


def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def get_best_span(ypi, yp2i):
    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        for j in range(len(ypif)):
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 * val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 * val2
    return ((best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)), float(max_val)


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs


def parse_args(path, config, **kwargs):
    source_dir = os.path.join(path, "download", "squad")
    target_dir = "data/squad"
    out_dir = "data/work"
    glove_dir = os.path.join(path, "data", "glove")
    argsdict = {
        'source_dir': source_dir,
        'target_dir': target_dir,
        'data_dir': target_dir,
        'glove_dir': glove_dir,
        'out_dir': out_dir,
        'shared_path': None,
        'train_ratio': 0.9,
        'glove_vec_size': 100,
        'glove_corpus': "6B",
        'mode': 'full',
        'tokenizer': "PTB",
        'split': True,
        'load': True,
        'finetune': False,
        'use_glove_for_unk': True,
        'known_if_glove': True,
        'debug': False,
        # performance thresholds
        'ques_size_th': 30,
        'num_sents_th': 8,
        'sent_size_th': 400,
        'para_size_th': 256,
        'word_size_th': 16,
        'char_count_th': 50,
        # merge all sentences in a paragraph
        'squash': False,
        'lower_word': True,
        # supervise only the answer sentence
        'single': False,
        # max | valid | semi
        'data_filter': 'max',
    }
    for key in kwargs.keys():
        argsdict[key] = kwargs[key]
    for key in config.__dict__.keys():
        if '__' in key or key == 'type':
            continue
        argsdict[key] = config.__dict__[key]
    return SQUAD_config(**argsdict)


def verify(args):
    e = os.path.exists
    j = os.path.join
    t = args.target_dir
    w = args.out_dir

    ddev, dtrain, dtest, sdev, strain, stest, out = \
        j(t, "data_dev.json"), j(t, "data_train.json"), j(t, "data_test.json"),\
        j(t, "shared_dev.json"), j(t, "shared_train.json"), j(t, "shared_test.json"),\
        j(w, "shared.json")

    files = [ddev, dtrain, dtest, sdev, strain, stest, out]
    nef = [f for f in files if not e(f)]
    if len(nef) == 0:
        return
    
    prepro(args)
    

def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    query, chquery, span, chspan, q2ctxt, ids, idxs = [], [], [], [], [], [], []
    ctxt, ctxt_sent, chctxt_sent = [], [], []
    answerss = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        ctxt_sent.append(xp)
        chctxt_sent.append(cxp)
        ctxt.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(ctxt_sent) - 1 == ai
            assert len(ctxt_sent[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1]-1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                query.append(qi)
                chquery.append(cqi)
                span.append(yi)
                chspan.append(cyi)
                q2ctxt.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

            if args.debug:
                break

#    word2vec_dict = get_word2vec(args, word_counter)
#    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'query': query, 'chquery': chquery, 'span': span, 'q2ctxt': q2ctxt, 'chspan': chspan,
            'idxs': idxs, 'ids': ids, 'answers': answers}
    shared = {'ctxt_sent': ctxt_sent, 'chctxt_sent': chctxt_sent, 'ctxt': ctxt,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter}
#              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)


def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")

    if not os.path.exists(shared_path):
        ref = False
    char_counter = shared['char_counter']
    shared['char2idx'] = {char: idx + 2 for idx, char in
                          enumerate(char for char, count in char_counter.items()
                                    if count > config.char_count_th)}
    #if not ref:
    #     #word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    #     word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
    #     if config.finetune:
    #         shared['word2idx'] = {word: idx + 2 for idx, word in
    #                               enumerate(word for word, count in word_counter.items()
    #                                         if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
    #     else:
    #         assert config.known_if_glove
    #         assert config.use_glove_for_unk
    #         shared['word2idx'] = {word: idx + 2 for idx, word in
    #                               enumerate(word for word, count in word_counter.items()
    #                                         if count > config.word_count_th and word not in word2vec_dict)}
    if not ref:
        NULL = "-NULL-"
        UNK = "-UNK-"
        #shared['word2idx'][NULL] = 0
        #shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    # if config.use_glove_for_unk:
    #     # create new word2idx and word2vec
    #     word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    #     # stoi of all words not in GloVe
    #     new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
    #     shared['new_word2idx'] = new_word2idx_dict
    #     offset = len(shared['word2idx']
    #     )
    #     # random embeddings of all non-GloVe words
    #     idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    #     # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
    #     new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    #     shared['new_emb_mat'] = new_emb_mat

    return data, shared, valid_idxs


def get_squad_data_filter(config):
    def data_filter(data_point, shared):
        assert shared is not None
        q2ctxt,  query, chquery, span = (data_point[key] for key in ('q2ctxt', 'query', 'chquery', 'span'))
        ctxt, chctxt = shared['ctxt_sent'], shared['chctxt_sent']
        if len(query) > config.ques_size_th:
            return False

        # x filter
        xi = ctxt[q2ctxt[0]][q2ctxt[1]]
        if config.squash:
            for start, stop in span:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th:
                    return False
            return True

        if config.single:
            for start, stop in span:
                if start[0] != stop[0]:
                    return False

        if config.data_filter == 'max':
            for start, stop in span:
                    if stop[0] >= config.num_sents_th:
                        return False
                    if start[0] != stop[0]:
                        return False
                    if stop[1] >= config.sent_size_th:
                        return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in span:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]:
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter

def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data, shared, idxs in data_sets:
        config.char_vocab_size = len(shared['char2idx'])
        for idx in idxs:
            q2ctxt = data['q2ctxt'][idx]
            query = data['query'][idx]
            sents = shared['ctxt_sent'][q2ctxt[0]][q2ctxt[1]]
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(query) > 0:
                config.max_ques_size = max(config.max_ques_size, len(query))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in query))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)
    
    #config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    #config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    if config.single:
        config.max_num_sents = 1
    if config.squash:
        config.max_sent_size = config.max_para_size
        config.max_num_sents = 1



class SQUAD(textdata.Dataset):
    """Loads the SQuAD QA dataset."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.question))

    def __init__(self, path, tier="train", fields=None, config=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        config = parse_args(path, config, **kwargs)
        verify(config)
        prefix = os.path.join(path, "data", "squad", tier)
        cache_file = prefix + "_examples"

        self.config = config
        other = Field()
        if fields is None:
            raise Exception('expected 3 fields')
        if len(fields) != 3:
            raise Exception('expected 3 fields')
        if not isinstance(fields[0], (tuple, list)):
            fields = [("context", fields[0]), ("chcontext", other), ("question", fields[0]), \
                      ("chquestion", other), ("answer", fields[1]), ("span", fields[2])]
            
        if os.path.exists(cache_file + ".npz"):
            print("fast loading {} examples".format(tier))
            d = npz_load(cache_file)
            examples, self.data, self.shared, self.idxs = d['examples'], d['data'], d['shared'], d['idxs']
            super(SQUAD, self).__init__(examples, fields, **kwargs)
            return

        print("slow loading {} examples".format(tier))
        qafilter = get_squad_data_filter(config)
        ref = True
        if tier == "train":
            ref = config.load

        data, shared, idxs = read_data(config, tier, ref, data_filter=qafilter)
        self.data, self.shared, self.idxs = data, shared, idxs

        examples = []
        querylist, chquerylist = data['query'], data['chquery']
        spanlist, answerlist = data['span'], data['answers']
        sentlist, chsentlist = shared['ctxt_sent'], shared['chctxt_sent']
        q2ctxt = data['q2ctxt']
        for idx in idxs:
            query, chquery = querylist[idx], chquerylist[idx]
            answer = answerlist[idx]
            ai, pi = q2ctxt[idx]
            answer = answerlist[idx]

            exlist = [[sent, chsent, query, chquery, answer, span] for sent, chsent, span in \
                  zip(sentlist[ai][pi], chsentlist[ai][pi], spanlist[idx])]
            for ex in exlist:
                examples.append(textdata.Example.fromlist(ex, fields))

        selfdict = {'data': data, 'shared': shared, 'idxs': idxs, 'examples': examples}
        npz_save(cache_file, selfdict)
        super(SQUAD, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, context_field, span_field, answer_field, root='.', config=None):
        fields = [context_field, span_field, answer_field]

        train, dev, test =  (cls(root, tier="train", fields=fields, config=config),
                             cls(root, tier="dev", fields=fields, config=config),
                             cls(root, tier="test", fields=fields, config=config))

        config = train.config
        datasets = [(train.data, train.shared, train.idxs),
                    (dev.data, dev.shared, dev.idxs),
                    (test.data, test.shared, test.idxs)]
        update_config(config, datasets)
        train.config, dev.config, test.config = config, config, config
        return train, dev, test
