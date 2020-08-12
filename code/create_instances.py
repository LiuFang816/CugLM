# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import collections
import json
import random
from create_data_corpus import *
# import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", 'training_token_corpus.txt',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("input_type_file", 'training_type_corpus.txt',
                    "Input type raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", 'training_token_type_instances.txt',
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("token_vocab_file", 'vocab_token.txt',
                    "The token vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("type_vocab_file", 'vocab_type.txt',
                    "The type vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 30,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 5,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, origin_tokens, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_types,
                 is_random_next):
        self.tokens = tokens
        self.origin_tokens = origin_tokens
        self.next_tokens = origin_tokens[1:] + ['[PAD]']
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.masked_lm_types = masked_lm_types

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, word2id, type_word2id, max_seq_length,
                                    max_predictions_per_seq, output_file):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = file_to_id(word2id, instance.tokens)
        lm_input_ids = file_to_id(word2id, instance.origin_tokens)
        lm_target_ids = file_to_id(word2id, instance.next_tokens)
        # input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        input_type_ids = file_to_id(type_word2id, instance.masked_lm_types)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            lm_input_ids.append(0)
            lm_target_ids.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        # assert len(input_type_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = file_to_id(word2id, instance.masked_lm_labels)

        # masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            input_type_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        # for Language model, no mask tokens
        features["lm_input_ids"] = create_int_feature(lm_input_ids)
        features["lm_target_ids"] = create_int_feature(lm_target_ids)

        features["input_type_ids"] = create_int_feature(input_type_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            # tf.logging.info("*** Example ***")
            # tf.logging.info("tokens: %s" % " ".join(
            #     [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, input_type_files, vocab, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    all_type_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    with open(input_files, 'r', encoding='utf-8') as f:
        tokendata = f.readlines()
    with open(input_type_files, 'r', encoding='utf-8') as f:
        typedata = f.readlines()
    assert len(tokendata) == len(typedata) # 53904364
    for i in tqdm(range(len(tokendata))):
        tokenline = tokendata[i].strip()
        typeline = typedata[i].strip()
        if not tokenline:
            all_documents.append([])
            all_type_documents.append([])
        else:
            tokens = json.loads(tokenline)
            types = json.loads(typeline)
            if tokens and types:
                all_documents[-1].append(tokens)
                all_type_documents[-1].append(types)


    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    all_type_documents = [x for x in all_type_documents if x]
    assert len(all_documents) == len(all_type_documents)
    rng.seed(FLAGS.random_seed)
    rng.shuffle(all_documents)
    rng.seed(FLAGS.random_seed)
    rng.shuffle(all_type_documents)

    vocab_words = list(vocab.keys())
    instances = []
    for fac in range(dupe_factor):
        print('dupe time: {}'.format(fac+1))
        for document_index in tqdm(range(len(all_documents))):
            instances.extend(
                create_instances_from_document(
                    all_documents, all_type_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, all_type_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]
    type_document = all_type_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_chunk_type = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        type_segment = type_document[i]
        current_chunk.append(segment)
        current_chunk_type.append(type_segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                types_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    types_a.extend(current_chunk_type[j])
                tokens_b = []
                types_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_type_document = all_type_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        types_b.extend(random_type_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                        types_b.extend(current_chunk_type[j])

                truncate_seq_pair(tokens_a, tokens_b, types_a, types_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                types = []
                segment_ids = []
                tokens.append("[CLS]")
                types.append("[CLS]")
                segment_ids.append(0)

                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                for t in types_a:
                    types.append(t)

                tokens.append("[SEP]")
                types.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                for t in types_b:
                    types.append(t)

                tokens.append("[SEP]")
                types.append("[SEP]")

                segment_ids.append(1)

                (output_tokens, masked_lm_positions,
                 masked_lm_labels, masked_lm_types) = create_masked_lm_predictions(
                    tokens, types, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=output_tokens,
                    origin_tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    masked_lm_types=masked_lm_types)
                instances.append(instance)

            current_chunk = []
            current_chunk_type = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label", "type"])


def create_masked_lm_predictions(tokens, types, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    assert len(tokens) == len(types)
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append(i)


    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []

    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if types[index] != '_':
            masked_token = "[MASK]"
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index], type=types[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    masked_lm_types = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
        masked_lm_types.append(p.type)

    return (output_tokens, masked_lm_positions, masked_lm_labels, masked_lm_types)



def truncate_seq_pair(tokens_a, tokens_b, types_a, types_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        trunc_types = types_a if len(types_a) > len(types_b) else types_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
            del trunc_types[0]
        else:
            trunc_tokens.pop()
            trunc_types.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = FLAGS.input_file
    input_type_files = FLAGS.input_type_file
    token_word2id, token_vocab_size = read_vocab(FLAGS.token_vocab_file)
    print(token_word2id['[PAD]'])
    print(token_vocab_size)
    type_word2id, type_vocab_size = read_vocab(FLAGS.type_vocab_file)
    print(type_word2id['[PAD]'])

    # for input_pattern in FLAGS.input_file.split(","):
    #     input_files.extend(tf.gfile.Glob(input_pattern))
    #
    # tf.logging.info("*** Reading from input files ***")
    # for input_file in input_files:
    #     tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, input_type_files, token_word2id, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_file = FLAGS.output_file
    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, token_word2id, type_word2id, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("input_type_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("token_vocab_file")
    flags.mark_flag_as_required("type_vocab_file")

    tf.app.run()
