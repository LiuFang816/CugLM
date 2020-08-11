# encoding: utf-8
import os
import json
import random
#from tqdm import tqdm
#from java_tokenizer import *
from collections import Counter
import six

# from bert_for_code.process_data.java_tokenizer import *

# project_path = 'E:\\data\\java_stactic\\Java_1516_format\\Java_1516'
# token_type_path = 'E:\data\java_stactic\java_token_type'
# project_dict = json.loads(open('project_file_dict.json', 'r').read())

def get_pre_train_projects():
    keys = list(project_dict.keys())
    random.seed(5)
    random.shuffle(keys)
    pre_train_projects = keys[:int(0.8 * len(keys))]  # 7771 projects; 789035 java files
    return pre_train_projects

# pre_train_projects = get_pre_train_projects()

def get_test_projects(testpath):
    projects = os.listdir(testpath)
    return projects

# test_projects = get_test_projects('E:\\代码\\static_java\\java_data\\cross_projects\\test')


def get_files(projects, project_dict):
    files = []
    for project in projects:
        files.extend(project_dict[project])
    random.seed(5)
    random.shuffle(files)
    return files

# files = get_files(test_projects, project_dict)


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


# print(len(pre_train_projects))
# files_num = 0
# for key in pre_train_projects:
#     files_num += len(project_dict[key])
# print(files_num)

# with open('pretrain_projects.json','w') as f:
#     f.write(json.dumps(pre_train_projects))

def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens, _, _, lineno = tokenize_java(text)
    lines = []
    pre_line = lineno[0]
    cur_line = []
    for i in range(len(lineno)):
        if lineno[i] == pre_line:
            cur_line.append(tokens[i])
        else:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
            else:
                lines.append(cur_line)
            cur_line = []
            pre_line = lineno[i]
            cur_line.append(tokens[i])
        if i == len(lineno) - 1:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
            else:
                lines.append(cur_line)
    return lines


def tokenize_file_withtype(file_path, token_type_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(file_path, 'r', encoding='utf-8') as f:
        textlines = f.readlines()

    token_types = json.loads(open(token_type_path).read())
    tokens, _, indexes, lineno = tokenize_java(text, textlines, need_index=True)
    types = []
    for i in range(len(indexes)):
        if str(indexes[i]) in token_types.keys():
            if token_types[str(indexes[i])][1] != '':
                types.append(token_types[str(indexes[i])][1][0].split('|')[0])
            else:
                types.append('_')
        else:
            types.append('_')
    assert len(tokens)==len(types)
    lines = []
    lines_type = []

    pre_line = lineno[0]
    cur_line = []
    cur_line_types = []

    for i in range(len(lineno)):
        if lineno[i] == pre_line:
            cur_line.append(tokens[i])
            cur_line_types.append(types[i])
        else:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
                lines_type[-1].append(cur_line_types[0])
            else:
                assert len(cur_line)==len(cur_line_types)
                lines.append(cur_line)
                lines_type.append(cur_line_types)

            cur_line = []
            cur_line_types = []
            pre_line = lineno[i]
            cur_line.append(tokens[i])
            cur_line_types.append(types[i])

        if i == len(lineno) - 1:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
                lines_type[-1].append(cur_line_types[0])
            else:
                assert len(cur_line)==len(cur_line_types)
                lines.append(cur_line)
                lines_type.append(cur_line_types)
    return lines, lines_type


# lines = tokenize_file('test.java')
# for line in lines:
#     print(line)

# files = get_files(pre_train_projects, project_dict)

def build_vocab(data, vocab_size=None, vocab_path=None):
    words = []
    for line in tqdm(data):
        words.extend(line)
    counter = Counter(words)
    # total = sum(counter.values())
    # print('total: {}'.format(total))
    if vocab_size:
        counter_pairs = counter.most_common(vocab_size - 5)
    else:
        counter_pairs = counter.most_common()
    words, values = list(zip(*counter_pairs))
    # print(len(words))
    # choosed_value = sum(values)
    # print(choosed_value)
    # print(choosed_value / total)
    # total_unk = total - choosed_value
    # print(total_unk*1.0/total)

    words = list(words)
    # words = ['[PAD]'] + ['[UNK]'] + ['[CLS]'] + ['[SEP]'] + ['[MASK]'] + words
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(words))


def read_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        # words = f.readlines()
        # words = [word.strip() for word in words]
        words = json.loads(f.read())
        words = ['[PAD]'] + ['[UNK]'] + ['[CLS]'] + ['[SEP]'] + ['[MASK]'] + words
        vocab_size = len(words)
        word_to_id = dict(zip(words, range(len(words))))
    return word_to_id, vocab_size


def file_to_id(word_to_id, data):
    for i in range(len(data)):
        data[i] = word_to_id[data[i]] if data[i] in word_to_id else word_to_id['[UNK]']
    return data

def create_training_corpus(pre_train_projects, project_dict, save_path):
    wf = open(save_path, 'w', encoding='utf-8')
    documents = []
    files = get_files(pre_train_projects, project_dict)
    for file in tqdm(files):
        try:
            code = tokenize_file(os.path.join(project_path, file))
            documents.append(code)
            for line in code:
                wf.write(json.dumps(line) + '\n')
            wf.write('\n')
        except:
            pass
    wf.close()
    return documents

def create_training_withtype_corpus(pre_train_projects, project_dict, token_save_path, type_save_path):
    wf_token = open(token_save_path, 'w', encoding='utf-8')
    wf_type = open(type_save_path, 'w', encoding='utf-8')
    documents = []
    documents_type = []
    files = get_files(pre_train_projects, project_dict)
    for file in tqdm(files):
        try:
            code, types = tokenize_file_withtype(os.path.join(project_path, file), os.path.join(token_type_path, file))
        except:
            continue

        assert len(code)==len(types)
        # documents.append(code)
        # documents_type.append(types)
        for i in range(len(code)):
            wf_token.write(json.dumps(code[i]) + '\n')
            wf_type.write(json.dumps(types[i]) + '\n')
        wf_token.write('\n')
        wf_type.write('\n')

    wf_token.close()
    wf_type.close()
    return documents, documents_type

# documents, documents_type = create_training_withtype_corpus(pre_train_projects, project_dict, 'training_token_corpus.txt', 'training_type_corpus.txt')
# documents, documents_type = create_training_withtype_corpus(test_projects, project_dict, 'test_token_corpus.txt', 'test_type_corpus.txt')

def save_vocab(data_path, size, path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        tokens = []
        for line in tqdm(data):
            line=line.strip()
            if line:
               tokens.append(json.loads(line))
        # print(tokens[:100])
        build_vocab(tokens, vocab_size=size, vocab_path=path)

# with open('large_training_token_corpus.txt','r') as f:
#     data = f.readlines()
#     print(len(data))
#     print(data[-10:])
# with open('large_training_type_corpus.txt','r') as f:
#     data = f.readlines()
#     print(len(data))
#     print(data[-10:])

# save_vocab('/data/liufang/static_java/java_bert/large_training_token_corpus.txt', 50000, '/data/liufang/static_java/java_bert/large_vocab_token.txt')
# save_vocab('/data/liufang/static_java/java_bert/large_training_type_corpus.txt', 50000, '/data/liufang/static_java/java_bert/large_vocab_type.txt')
# word_to_id, vocab_size = read_vocab('large_vocab_token.txt')
# print(word_to_id['[PAD]'])
