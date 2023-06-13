import random
import math
import torch
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers import BertTokenizer, BertTokenizerFast
import warnings
import copy


class DataCollatorMixin:
    def __call__(self, features, return_tensors="pt"):
        if return_tensors == "pt":
            return self.torch_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForGeoQueryModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    geo_aware_open: bool
    order_open: bool
    cl_open: bool
    nsp_open: bool
    triplet_open: bool
    hard_negative: bool
    geohash_open: bool
    mlm: bool = True
    mlm_open: bool = True
    norm_mlm_probability: float = 0.15
    single_geo_mlm_probability: float = 0.30
    cover_geo_mlm_probability: float = 0.50
    pos_shuffle_probability: float = 0.15
    geohash_bits: int = 6
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if isinstance(examples[0], Mapping):
            if self.cl_open or self.triplet_open:
                query_input_ids = [e["input_ids"][0] for e in examples]
                combine_input_ids = [e["input_ids"][1] for e in examples]
                aug_input_ids = [e["input_ids"][2] for e in examples]
                if self.hard_negative or self.triplet_open:
                    neg_input_ids = [e["input_ids"][3] for e in examples]
                    batch_input = _torch_collate_batch(
                        query_input_ids + combine_input_ids + aug_input_ids + neg_input_ids, self.tokenizer,
                        pad_to_multiple_of=self.pad_to_multiple_of)
                else:
                    batch_input = _torch_collate_batch(query_input_ids + combine_input_ids + aug_input_ids,
                                                       self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                bsz = len(query_input_ids)
                query_batch_input = copy.deepcopy(batch_input[bsz:bsz * 2])
                order_batch_input = copy.deepcopy(batch_input[:bsz])
                mask_batch_input = batch_input[bsz:bsz * 2]
                if self.hard_negative or self.triplet_open:
                    pos_batch_input = batch_input[bsz * 2:bsz * 3]
                    neg_batch_input = batch_input[bsz * 3:]
                else:
                    pos_batch_input = batch_input[bsz * 2:]
                    neg_batch_input = None
                for e in examples:
                    for key in e:
                        if key in special_keys:
                            e[key] = e[key][1]
                        else:
                            e[key] = e[key]

            elif self.nsp_open:
                query_input_ids = [e["input_ids"][0] for e in examples]
                combine_input_ids = [e["input_ids"][1] for e in examples]
                pos_input_ids = [e["input_ids"][2] for e in examples]
                neg_input_ids = [e["input_ids"][3] for e in examples]
                tokens_type_ids = [e["token_type_ids"][0] for e in examples] + [e["token_type_ids"][1] for e in
                                                                                examples] + [e["token_type_ids"][2] for
                                                                                             e in examples] + [
                                      e["token_type_ids"][3] for e in examples]

                bsz = len(query_input_ids)
                batch_input = _torch_collate_batch(query_input_ids + combine_input_ids + pos_input_ids + neg_input_ids,
                                                   self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                tokens_type_ids = _torch_collate_batch(tokens_type_ids, self.tokenizer,
                                                       pad_to_multiple_of=self.pad_to_multiple_of)
                query_batch_input = copy.deepcopy(batch_input[bsz:bsz * 2])
                order_batch_input = copy.deepcopy(batch_input[:bsz])
                mask_batch_input = batch_input[bsz:bsz * 2]
                pos_batch_input = batch_input[bsz * 2:bsz * 3]
                neg_batch_input = batch_input[bsz * 3:]
                for e in examples:
                    for key in e:
                        if key in special_keys:
                            e[key] = e[key][1]
                        else:
                            e[key] = e[key]

            else:
                query_input_ids = [e["input_ids"][0] for e in examples]
                combine_input_ids = [e["input_ids"][1] for e in examples]
                bsz = len(query_input_ids)
                batch_input = _torch_collate_batch(query_input_ids + combine_input_ids, self.tokenizer,
                                                   pad_to_multiple_of=self.pad_to_multiple_of)
                query_batch_input = copy.deepcopy(batch_input[bsz:bsz * 2])
                order_batch_input = copy.deepcopy(batch_input[:bsz])
                mask_batch_input = batch_input[bsz:bsz * 2]
                pos_batch_input = None
                neg_batch_input = None
                for e in examples:
                    for key in e:
                        if key in special_keys:
                            e[key] = e[key][1]
                        else:
                            e[key] = e[key]
        else:
            raise ValueError(
                "Wrong tokenization process."
            )

        mask_labels = []
        org_orders = []
        geohash_labels = [[] for _ in range(self.geohash_bits)]

        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For geography tokens, we need extra inf to mark sub-word, e.g., [杭州]-> [杭，##州] or [杭，###州]
            # '##' is single-geo, and '###' is cover-geo

            if "order_ref" in e:
                org_orders.append(tolist(e["order_ref"]))

            if "geohash_label" in e:
                for i in range(self.geohash_bits):
                    # if e["geohash_label"][i] != '0':
                    geohash_labels[i].append(torch.tensor(int(e["geohash_label"][i])))
                    # else:
                    #     geohash_labels[i].append(torch.tensor(-100))

            # if "implicit_info" in e:
            #     implicit_info.append(e["implicit_info"])

            if self.geo_aware_open:
                if "single_geo_ref" in e and "cover_geo_ref" in e:
                    single_ref_pos = tolist(e["single_geo_ref"])
                    cover_ref_pos = tolist(e["cover_geo_ref"])
                    len_seq = len(e["input_ids"])
                    for i in range(len_seq):
                        if i in single_ref_pos:
                            ref_tokens[i] = "##" + ref_tokens[i]
                        if i in cover_ref_pos:
                            ref_tokens[i] = "###" + ref_tokens[i]
                mask_labels.append(self._whole_word_mask(ref_tokens))
            else:
                mask_labels.append(self._whole_word_mask_org(ref_tokens))

        if self.mlm_open:
            batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, self.nsp_open,
                                              pad_to_multiple_of=self.pad_to_multiple_of,
                                              pad_to_specifid_length=len(mask_batch_input[0]))
            mlm_batch_input, mlm_labels = self.torch_mask_tokens(mask_batch_input, batch_mask)
        else:
            mlm_batch_input, mlm_labels = mask_batch_input, None

        if self.order_open:
            order_batch_input, tokens_order_labels, phrases_order_labels = self._replace_position_ids(order_batch_input,
                                                                                                      org_orders)
        else:
            tokens_order_labels, phrases_order_labels = None, None
        order_input_attention_mask = order_batch_input.bool().long()
        for i in range(self.geohash_bits):
            geohash_labels[i] = torch.tensor(geohash_labels[i])

        if self.geohash_open:
            batch_geohash_labels = geohash_labels
            # print(batch_geohash_labels.shape)
            # assert 1==2
        else:
            batch_geohash_labels = None

        nsp_batch_input = None
        nsp_labels = None
        nsp_attention_mask = None

        if self.cl_open or self.triplet_open:
            assert query_batch_input.shape == mlm_batch_input.shape == pos_batch_input.shape
            query_input_attention_mask = query_batch_input.bool().long()
            mlm_input_attention_mask = mlm_batch_input.bool().long()
            pos_input_attention_mask = pos_batch_input.bool().long()

            if self.hard_negative or self.triplet_open:
                assert pos_batch_input.shape == neg_batch_input.shape
                neg_input_attention_mask = neg_batch_input.bool().long()
            else:
                neg_input_attention_mask = None

        elif self.nsp_open:
            assert query_batch_input.shape == mlm_batch_input.shape == pos_batch_input.shape == neg_batch_input.shape
            query_input_attention_mask = query_batch_input.bool().long()
            mlm_input_attention_mask = mlm_batch_input.bool().long()
            pos_input_attention_mask = pos_batch_input.bool().long()
            neg_input_attention_mask = neg_batch_input.bool().long()
            nsp_labels = torch.tensor(
                [1 for _ in range(len(query_batch_input))] + [0 for _ in range(len(query_batch_input))])
            nsp_batch_input = torch.cat((pos_batch_input, neg_batch_input), 0)
            nsp_batch_tokens_type_ids = tokens_type_ids[bsz * 2:]
            nsp_zip = [(x, y, z) for x, y, z in zip(nsp_batch_input, nsp_labels, nsp_batch_tokens_type_ids)]
            random.shuffle(nsp_zip)
            nsp_batch_input = torch.cat([x[0].unsqueeze(0) for x in nsp_zip], 0)
            nsp_labels = torch.cat([x[1].unsqueeze(0) for x in nsp_zip], 0)
            nsp_batch_tokens_type_ids = torch.cat([x[2].unsqueeze(0) for x in nsp_zip], 0)
            tokens_type_ids = torch.cat((tokens_type_ids[:bsz * 2], nsp_batch_tokens_type_ids), 0)
            nsp_attention_mask = nsp_batch_input.bool().long()
        else:
            assert query_batch_input.shape == mlm_batch_input.shape
            query_input_attention_mask = query_batch_input.bool().long()
            mlm_input_attention_mask = mlm_batch_input.bool().long()
            pos_input_attention_mask = None
            neg_input_attention_mask = None
        # for i in range(bsz):
        #     print(query_batch_input[i])
        #     print(self.tokenizer.decode(query_batch_input[i]))
        #     print(self.tokenizer.decode(order_batch_input[i]))
        # print(self.tokenizer.decode(nsp_batch_input[i]))
        # print(mlm_batch_input[i])
        # print(mlm_labels[i])
        # print(phrases_order_labels[i])
        # print(tokens_order_labels[i])
        # print(pos_batch_input[i])
        # print(self.tokenizer.decode(pos_batch_input[i]))
        if not self.nsp_open:
            tokens_type_ids = None
        return {"input_ids": query_batch_input, "token_type_ids": tokens_type_ids, "mlm_input_ids": mlm_batch_input,
                "order_input_ids": order_batch_input, "nsp_input_ids": nsp_batch_input, "mlm_labels": mlm_labels,
                "tokens_order_labels": tokens_order_labels, "phrases_order_labels": phrases_order_labels,
                "geohash_labels": batch_geohash_labels,
                "nsp_labels": nsp_labels,
                "pos_input_ids": pos_batch_input, "neg_input_ids": neg_batch_input,
                "query_input_attention_mask": query_input_attention_mask,
                "mlm_input_attention_mask": mlm_input_attention_mask,
                "pos_input_attention_mask": pos_input_attention_mask,
                "neg_input_attention_mask": neg_input_attention_mask,
                "order_input_attention_mask": order_input_attention_mask,
                "nsp_input_attention_mask": nsp_attention_mask}

    def _replace_position_ids(self, input_ids, order_ref):

        bsz, seq_len = input_ids.shape

        # shuffle inpuy_ids
        shuffled_inputs = []
        phrases_order_labels = []
        tokens_order_labels = []

        for bsz_id in range(bsz):

            # create query order label

            phrases_order_label = self.flatten_order_list(
                [[i for _ in range(len(order_ref[bsz_id][i]))] for i in range(len(order_ref[bsz_id]))])
            tokens_order_label = self.flatten_order_list(
                [[idx for idx in range(len(order_ref[bsz_id][i]))] for i in range(len(order_ref[bsz_id]))])
            phrases_order_label[0] = -100
            tokens_order_label[0] = -100
            assert len(phrases_order_label) == len(tokens_order_label)

            phrases_order_label += [-100 for _ in range(seq_len - len(phrases_order_label))]
            tokens_order_label += [-100 for _ in range(seq_len - len(tokens_order_label))]

            phrases_order_label = torch.tensor(phrases_order_label)
            tokens_order_label = torch.tensor(tokens_order_label)

            import random
            # if random.random() < self.pos_shuffle_probability:
            #    random.shuffle(order_ref[bsz_id])
            for sub_list in order_ref[bsz_id]:
                if random.random() < self.pos_shuffle_probability:
                    random.shuffle(sub_list)
            random.shuffle(order_ref[bsz_id])
            indexes = self.flatten_order_list(order_ref[bsz_id])
            # for idx, order in enumerate(indexes):
            #     order_label[idx] = order
            rest_indexes = list(range(len(indexes), seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_inputs.append(torch.index_select(input_ids[bsz_id], 0, torch.tensor(total_indexes).to(
                device=input_ids.device)).unsqueeze(0))

            phrases_order_labels.append(torch.index_select(phrases_order_label, 0, torch.tensor(total_indexes).to(
                device=input_ids.device)).unsqueeze(0))

            tokens_order_labels.append(torch.index_select(tokens_order_label, 0, torch.tensor(total_indexes).to(
                device=input_ids.device)).unsqueeze(0))

        return torch.cat(shuffled_inputs, 0), torch.cat(tokens_order_labels, 0), torch.cat(phrases_order_labels, 0)

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )
        single_cand_indexes = []
        cover_cand_indexes = []
        # Place holder
        norm_cand_indexes = [0 for i in range(len(input_tokens))]

        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if len(cover_cand_indexes) >= 1 and token.startswith("###"):
                cover_cand_indexes[-1].append(i)
            else:
                cover_cand_indexes.append([i])

            if len(single_cand_indexes) >= 1 and token.startswith("##") and not token.startswith("###"):
                single_cand_indexes[-1].append(i)
            else:
                single_cand_indexes.append([i])

        final_single_cand_indexes = []
        final_cover_cand_indexes = []

        for i, sub_list in enumerate(single_cand_indexes):
            if len(sub_list) == 1:
                norm_cand_indexes[sub_list[0]] = 1
            else:
                final_single_cand_indexes.append(sub_list)
                for idx in sub_list:
                    norm_cand_indexes[idx] = 0

        for i, sub_list in enumerate(cover_cand_indexes):
            if len(sub_list) == 1:
                norm_cand_indexes[sub_list[0]] &= 1
            else:
                final_cover_cand_indexes.append(sub_list)
                for idx in sub_list:
                    norm_cand_indexes[idx] &= 0
        final_norm_cand_indexes = [[i] for i in range(len(norm_cand_indexes)) if norm_cand_indexes[i] == 1]
        # final_norm_cand_indexes = [[i] for i in range(len(norm_cand_indexes))]
        random.shuffle(final_norm_cand_indexes)
        random.shuffle(final_single_cand_indexes)
        random.shuffle(final_cover_cand_indexes)

        norm_num_to_predict = min(max_predictions,
                                  math.ceil(cal_list_size(final_norm_cand_indexes) * self.norm_mlm_probability))
        single_num_to_predict = min(max_predictions, math.ceil(
            cal_list_size(final_single_cand_indexes) * self.single_geo_mlm_probability))
        cover_num_to_predict = min(max_predictions,
                                   math.ceil(cal_list_size(final_cover_cand_indexes) * self.cover_geo_mlm_probability))

        covered_indexes = set()

        self.pos_mask(covered_indexes, final_norm_cand_indexes, norm_num_to_predict)
        self.pos_mask(covered_indexes, final_single_cand_indexes, single_num_to_predict)
        self.pos_mask(covered_indexes, final_cover_cand_indexes, cover_num_to_predict)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

        return mask_labels

    def _whole_word_mask_org(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.norm_mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        try:
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        except:
            print(inputs)
            print(special_tokens_mask)
            print(torch.tensor(special_tokens_mask, dtype=torch.bool).shape)  # 45
            print(probability_matrix.shape)  # 36
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def pos_mask(self, covered_indexes, cand_indexes, num_to_predict):

        masked_lms = []

        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        # return covered_indexes, masked_lms

    def flatten_order_list(self, order_label):
        res = [0]
        for sub_list in order_label:
            for unit in sub_list:
                res.append(unit)
        return res


def _torch_collate_batch(examples, tokenizer, nsp_open=False, pad_to_multiple_of: Optional[int] = None,
                         pad_to_specifid_length: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
            pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0) and not nsp_open:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)

    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    if pad_to_specifid_length:
        max_length = max(max_length, pad_to_specifid_length)

    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def cal_list_size(x):
    total_len = 0
    for l in x:
        total_len += len(l)
    return total_len