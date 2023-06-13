import argparse
import json
import os
from typing import List

from tqdm.contrib import tzip
from transformers.models.bert.tokenization_bert import BertTokenizer
from tqdm.autonotebook import tqdm
import random

random.seed(7)

hashtag_map = {'*': 0,
               '0': 1,
               '1': 2,
               '2': 3,
               '3': 4,
               '4': 5,
               '5': 6,
               '6': 7,
               '7': 8,
               '8': 9,
               '9': 10,
               'b': 11,
               'c': 12,
               'd': 13,
               'e': 14,
               'f': 15,
               'g': 16,
               'h': 17,
               'j': 18,
               'k': 19,
               'm': 20,
               'n': 21,
               'p': 22,
               'q': 23,
               'r': 24,
               's': 25,
               't': 26,
               'u': 27,
               'v': 28,
               'w': 29,
               'x': 30,
               'y': 31,
               'z': 32}


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def get_chinese_word(tokens: List[str]):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def geo_tokenizer(examples):
    ret_data = []
    new_data = []
    for example in examples:
        try:
            query, item, query_geo, item_geo, query_tag, term, hash_tag, similar_query = example.split('\t')
            query_tag = query_tag.split(";")
            assert query == "".join(x for x in query_tag)
            new_data.append(example)
        except:
            # print("< " + example + " > is invalid and it will be deleted")
            continue
        query_split = " ".join(x for x in list(query))
        item_split = " ".join(x for x in list(item))
        query_geo = set([x for x in query_geo.split(';') if x in query])
        query_geo.discard("none")
        item_geo = set([x for x in item_geo.split(';') if x in item])
        item_geo.discard("none")
        term_parse = json.loads(term)
        implicit_info = []
        query_with_implicit = ""
        for idx, unit in enumerate(term_parse):
            content = ["{}".format(unit["term"]), "[{}]".format(unit["second_cate"])]
            implicit_info.append(content)
            # if unit["second_cate"] == 'division2poi' or unit["second_cate"] == 'poiname2poi':
            #     query_geo.add(unit["term"])
            #     if unit["term"] in item:
            #         item_geo.add(unit["term"])
        random.shuffle(implicit_info)
        for unit in implicit_info:
            query_with_implicit += "".join(x for x in unit)
        # query_with_implicit += "[SEP]" + item

        hash_tag_label = [hashtag_map[x] for x in hash_tag.split(' ')]

        for geo in query_geo:
            query_split = query_split.replace(" ".join(x for x in list(geo)), geo)
        for geo in item_geo:
            item_split = item_split.replace(" ".join(x for x in list(geo)), geo)
        combined_token = query_split.split(" ")
        combined_token.extend(['[SEP]'])
        combined_token.extend(item_split.split(" "))
        combine_str = "".join(x for x in combined_token)
        query_str = "".join(x for x in query_tag)
        ret_data.append({
            "combine": combine_str,
            "query": query_str,
            "item": item_split.split(" "),
            "query_geo": query_geo,
            "item_geo": item_geo,
            "query_tag": query_tag,
            "query_aug": query_with_implicit,
            "hash_tag_label": hash_tag_label,
            "similar_query": similar_query,
        })

    return ret_data, new_data


def add_sub_symbol_for_geo(bert_tokens: List[str], query_geo_set: set(), item_geo_set: set()):
    if not query_geo_set and not item_geo_set:
        return bert_tokens
    elif not query_geo_set:
        max_word_len = max([len(w) for w in item_geo_set])
    elif not item_geo_set:
        max_word_len = max([len(w) for w in query_geo_set])
    else:
        max_word_len = max(max([len(w) for w in query_geo_set]), max([len(w) for w in item_geo_set]))
    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start: start + i])
                if whole_word in query_geo_set and whole_word in item_geo_set:
                    for j in range(start + 1, start + i):
                        # '##' is single-geo, and '###' is cover-geo
                        bert_word[j] = "###" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
                elif whole_word in query_geo_set or whole_word in item_geo_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break

        if single_word:
            start += 1

    return bert_word


def add_sub_symbol_for_poi(bert_tokens: List[str], query_poi_set: set()):
    if not query_poi_set:
        return bert_tokens
    max_word_len = max([len(w) for w in query_poi_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)

    while start < end:
        single_word = True
        l = min(end - start, max_word_len)
        for i in range(l, 1, -1):
            whole_word = "".join(bert_word[start: start + i])

            if whole_word in query_poi_set:
                for j in range(start + 1, start + i):
                    bert_word[j] = "##" + bert_word[j]
                start = start + i
                single_word = False
                break

        if single_word:
            start += 1

    return bert_word


def prepare_ref(lines: List[str], bert_tokenizer: BertTokenizer):
    query_geo = []
    item_geo = []
    query = []
    query_tag = []
    query_aug = []
    new_data = []
    geo_label = []
    similar_query = []

    print("------Geo and Poi Tokenizing------")
    for i in tqdm(range(0, len(lines), 100)):
        ret, data = geo_tokenizer(lines[i: i + 100])
        query_geo.extend([x['query_geo'] for x in ret])
        item_geo.extend([x['item_geo'] for x in ret])
        new_data.extend([x['combine'] for x in ret])
        query.extend([x['query'] for x in ret])
        query_tag.extend([x['query_tag'] for x in ret])
        query_aug.extend([x['query_aug'] for x in ret])
        geo_label.extend([x['hash_tag_label'] for x in ret])
        similar_query.extend([x['similar_query'] for x in ret])

    assert len(new_data) == len(query_geo) == len(item_geo) == len(query_tag) == len(query_aug) == len(
        geo_label) == len(similar_query)

    bert_res = []
    bert_query_res = []

    print("------BERT Tokenizing------")

    for i in tqdm(range(0, len(new_data), 100)):
        text = [line for line in new_data[i:i + 100]]
        res = bert_tokenizer(text, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)
        bert_res.extend(res["input_ids"])

    for i in tqdm(range(0, len(query), 100)):
        text = [line for line in query[i:i + 100]]
        res = bert_tokenizer(text, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)
        bert_query_res.extend(res["input_ids"])

    assert len(bert_res) == len(new_data) == len(bert_query_res)

    single_ref_ids = []
    cover_ref_ids = []

    print("------Geo Ref Building------")
    for input_ids, geo_q, geo_i in tzip(bert_res, query_geo, item_geo):
        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        input_tokens = add_sub_symbol_for_geo(input_tokens, geo_q, geo_i)
        single_ref_id = []
        cover_ref_id = []
        # '##' is single-geo, and '###' is cover-geo
        for i, token in enumerate(input_tokens):
            if token[:3] == "###":
                clean_token = token[3:]
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    cover_ref_id.append(i)
            elif token[:2] == "##":
                clean_token = token[2:]
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    single_ref_id.append(i)

        single_ref_ids.append(single_ref_id)
        cover_ref_ids.append(cover_ref_id)

    order_ref_ids = []

    print("------Order Ref Building------")
    for input_ids, poi_token in tzip(bert_query_res, query_tag):

        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        input_tokens = add_sub_symbol_for_poi(input_tokens, poi_token)
        # if input_ids == [101, 6656, 1730, 128, 3189, 3952, 6205, 5966, 102]:
        #     print(input_tokens,poi_token)
        order_ref_id = []
        tmp = []
        for i, token in enumerate(input_tokens):
            if token[:2] == "##":
                clean_token = token[2:]
                # if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                tmp.append(i)
            elif token not in ["[CLS]", "[SEP]"]:
                if len(tmp) != 0:
                    order_ref_id.append(tmp)
                    tmp = []
                tmp.append(i)
            if i == len(input_tokens) - 1 and len(tmp) != 0:
                order_ref_id.append(tmp)
        order_ref_ids.append(order_ref_id)
    try:
        assert len(single_ref_ids) == len(cover_ref_ids) == len(order_ref_ids) == len(bert_res) == len(new_data) == len(
            geo_label) == len(similar_query)
    except:
        print(len(single_ref_ids))
        print(len(cover_ref_ids))
        print(len(order_ref_ids))
        print(len(bert_res))
        print(len(new_data))
        exit(0)

    return single_ref_ids, cover_ref_ids, order_ref_ids, new_data, query_aug, query, geo_label, similar_query


def split_train_val(data, val_split_percentage):
    if val_split_percentage < 0 or val_split_percentage > 1:
        raise ValueError(f"Invalid '{val_split_percentage}' which should be in '[0,1]' ")

    new_data = [(r, s, t, u, v, w, x, y) for r, s, t, u, v, w, x, y in data]
    random.shuffle(new_data)

    train_sep_num = int(len(new_data) * val_split_percentage)

    val_data = new_data[:train_sep_num]
    train_data = new_data[train_sep_num:]

    return train_data, val_data


def main(args):
    # For Chinese (Ro)Bert, the best result is from : RoBERTa-wwm-ext (https://github.com/ymcui/Chinese-BERT-wwm)
    # If we want to fine-tune these model, we have to use same tokenizer : LTP (https://github.com/HIT-SCIR/ltp)
    with open(args.file_name, "r", encoding="utf-8") as f:
        data = f.readlines()
    # with open('geo_pair.json', encoding='utf8') as f:
    #     geo_pair = json.load(f)
    data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]
    print("Corpus data size is " + str(len(data)))
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert)

    single_ref_ids, cover_ref_ids, order_ref_ids, new_data, query_aug, query, geo_label, similar_query = prepare_ref(
        data, bert_tokenizer)
    print("Filted data size is " + str(len(new_data)))

    train_data, val_data = split_train_val(
        zip(single_ref_ids, cover_ref_ids, order_ref_ids, new_data, query_aug, query, geo_label, similar_query),
        args.val_split_perentage)

    train_single_ref_ids = [x[0] for x in train_data]
    train_cover_ref_ids = [x[1] for x in train_data]
    train_order_ref_ids = [x[2] for x in train_data]
    train_text = [x[3] for x in train_data]
    train_aug = [x[4] for x in train_data]
    train_query = [x[5] for x in train_data]
    train_geo_label = [x[6] for x in train_data]
    train_similar_query = [x[7] for x in train_data]

    val_single_ref_ids = [x[0] for x in val_data]
    val_cover_ref_ids = [x[1] for x in val_data]
    val_order_ref_ids = [x[2] for x in val_data]
    val_text = [x[3] for x in val_data]
    val_aug = [x[4] for x in val_data]
    val_query = [x[5] for x in val_data]
    val_geo_label = [x[6] for x in val_data]
    val_similar_query = [x[7] for x in val_data]

    prefix = args.save_path + "/"

    if not os.path.exists(prefix + "train"):
        os.mkdir(prefix + "train")
    if not os.path.exists(prefix + "val"):
        os.mkdir(prefix + "val")
    if not os.path.exists(prefix + "ref"):
        os.mkdir(prefix + "ref")

    with open(prefix + "ref/train_single_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in train_single_ref_ids]
        f.writelines(data)
    with open(prefix + "ref/val_single_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in val_single_ref_ids]
        f.writelines(data)

    with open(prefix + "ref/train_cover_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in train_cover_ref_ids]
        f.writelines(data)
    with open(prefix + "ref/val_cover_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in val_cover_ref_ids]
        f.writelines(data)

    with open(prefix + "ref/train_order_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in train_order_ref_ids]
        f.writelines(data)
    with open(prefix + "ref/val_order_ref.txt", "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in val_order_ref_ids]
        f.writelines(data)

    with open(prefix + "ref/train_geo_label_ref.txt", "w", encoding="utf-8") as f:
        for data in train_geo_label:
            f.write(' '.join(str(x) for x in data) + '\n')
    with open(prefix + "ref/val_geo_label_ref.txt", "w", encoding="utf-8") as f:
        for data in val_geo_label:
            f.write(' '.join(str(x) for x in data) + '\n')

    with open(prefix + "train/" + "corpus_processed.txt", "w", encoding="utf-8") as f:
        for org, aug in zip(train_text, train_aug):
            f.write(org + "\t" + aug + "\n")

    with open(prefix + "val/" + "corpus_processed.txt", "w", encoding="utf-8") as f:
        for org, aug in zip(val_text, val_aug):
            f.write(org + "\t" + aug + "\n")

    with open(prefix + "train/" + "corpus_processed_with_similar_query.txt", "w", encoding="utf-8") as f:
        for org, sq in zip(train_text, train_similar_query):
            f.write(org + "\t" + sq + "\n")

    with open(prefix + "val/" + "corpus_processed_with_similar_query.txt", "w", encoding="utf-8") as f:
        for org, sq in zip(val_text, val_similar_query):
            f.write(org + "\t" + sq + "\n")

    with open(prefix + "train_query.txt", "w", encoding="utf-8") as f:
        for org in train_query:
            f.write(org + "\n")

    with open(prefix + "val_query.txt", "w", encoding="utf-8") as f:
        for org in val_query:
            f.write(org + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare_geo_and_order_ref")
    parser.add_argument(
        "--file_name",
        type=str,
        default="data/corpus.txt",
        help="file need process, same as training data in lm",
    )
    parser.add_argument("--bert", type=str, default="bert-base-chinese",
                        help="resources for Bert tokenizer")
    parser.add_argument("--max_seq_length", type=int, default="128", help="max seq length")
    parser.add_argument("--val_split_perentage", type=float, default=0.01,
                        help="The percentage of the corpus used as validation set")
    parser.add_argument("--save_path", type=str, default="data",
                        help="path dir to save res")

    args = parser.parse_args()
    # args = parser.parse_known_args()[0]
    main(args)