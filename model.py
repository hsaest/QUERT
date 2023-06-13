import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel, BertOnlyNSPHead
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import logging, ModelOutput
from transformers.activations import ACT2FN
import torch.nn.functional as F
import torch.distributed as dist
from ernie import ErnieModel, ErniePreTrainedModel


def bert_add_tokens(tokenizer, tokens_file):
    tokens_list = open(tokens_file, 'r', encoding='utf-8').read().strip().split('\n')
    new_special_tokens_dict = {"additional_special_tokens": tokens_list}
    tokenizer.add_special_tokens(new_special_tokens_dict)

    return tokenizer


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'mean': average of the last layers' hidden states at each token.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "mean"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs

        if self.pooler_type == "cls":
            return last_hidden[:, 0]
        elif self.pooler_type == "mean":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
        # distance_positive = (anchor - positive).pow(2).sum(1)
        # distance_negative = (anchor - negative).pow(2).sum(1)
        distance_positive = self.cos(anchor, positive)
        distance_negative = self.cos(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses, losses.mean()


class TripletNet(nn.Module):
    def __init__(self, config):
        super(TripletNet, self).__init__()

        self.hidden_size = config.hidden_size
        self.dropout = config.hidden_dropout_prob

        self.embedding_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, x1, x2, x3):
        output1 = F.normalize(self.embedding_net(x1), p=2, dim=-1)
        output2 = F.normalize(self.embedding_net(x2), p=2, dim=-1)
        output3 = F.normalize(self.embedding_net(x3), p=2, dim=-1)
        return output1, output2, output3

    def get_embedding(self, encoder_feature):
        return self.embedding_net(encoder_feature)


@dataclass
class QueryBertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    order_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    triplet_loss: Optional[torch.FloatTensor] = None
    geohash_loss: Optional[torch.FloatTensor] = None
    nsp_loss: Optional[torch.FloatTensor] = None
    triplet_logits: torch.FloatTensor = None
    contrastive_logits: torch.FloatTensor = None
    prediction_logits: torch.FloatTensor = None
    tokens_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class QueryBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.config = config
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        self.phrases_order_prediction = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            ACT2FN[config.hidden_act],
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.phrases_order_size)
        )

        self.tokens_order_prediction = nn.Linear(config.phrases_order_size, config.tokens_order_size)

        if self.config.geohash_open:
            self.geohash_block = nn.ModuleList()
            for _ in range(config.hash_bits):
                self.geohash_block.append(
                    nn.Linear(config.hidden_size, config.hash_size)
                )

    def forward(self, combine_output, query_output, pooled_output=None):
        prediction_scores = self.predictions(combine_output)
        phrases_scores = self.phrases_order_prediction(query_output)
        tokens_scores = self.tokens_order_prediction(phrases_scores)
        if self.config.geohash_open:
            geohash_logits_store = []
            for idx in range(self.config.hash_bits):
                geohash_logits_store.append(self.geohash_block[idx](pooled_output).unsqueeze(0))
            # print(geohash_scores.shape)
            # assert 1==2
            return prediction_scores, phrases_scores, tokens_scores, geohash_logits_store
        else:
            return prediction_scores, phrases_scores, tokens_scores


class QueryBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = QueryBertPreTrainingHeads(config)
        self.config = config
        # Initialize weights and apply final processing

        if config.cl_open:
            self.pooler = Pooler(config.pooler_type)
            self.sim = Similarity(config.temp)
            if config.pooler_type == 'cls':
                self.mlp = MLPLayer(config)

        elif config.triplet_open:
            self.pooler = Pooler(config.pooler_type)
            self.sim = TripletLoss(config.margin)
            if config.pooler_type == 'cls':
                self.triplet_net = TripletNet(config)

        elif config.nsp_open:
            self.nsp_relationship = nn.Linear(config.hidden_size, 2)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            mlm_input_ids: Optional[torch.Tensor] = None,
            query_input_attention_mask: Optional[torch.Tensor] = None,
            order_input_ids: Optional[torch.Tensor] = None,
            nsp_input_ids: Optional[torch.Tensor] = None,
            mlm_input_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            mlm_labels: Optional[torch.Tensor] = None,
            tokens_order_labels: Optional[torch.Tensor] = None,
            nsp_labels: Optional[torch.Tensor] = None,
            phrases_order_labels: Optional[torch.Tensor] = None,
            geohash_labels: Optional[List[torch.Tensor]] = None,
            pos_input_ids: Optional[torch.Tensor] = None,
            neg_input_ids: Optional[torch.Tensor] = None,
            pos_input_attention_mask: Optional[torch.Tensor] = None,
            neg_input_attention_mask: Optional[torch.Tensor] = None,
            order_input_attention_mask: Optional[torch.Tensor] = None,
            nsp_input_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QueryBertForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bsz = input_ids.shape[0]
        if pos_input_ids is not None and pos_input_attention_mask is not None:

            if self.config.hard_negative or self.config.triplet_open:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, pos_input_ids, neg_input_ids), 0)
                attention_mask = torch.cat(
                    (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask,
                     pos_input_attention_mask,
                     neg_input_attention_mask), 0)

            elif self.config.nsp_open:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, nsp_input_ids), 0)
                attention_mask = torch.cat((mlm_input_attention_mask, order_input_attention_mask,
                                            query_input_attention_mask, nsp_input_attention_mask), 0)
                token_type_ids = torch.cat((token_type_ids[bsz:bsz * 2], token_type_ids[bsz:bsz * 2],
                                            token_type_ids[bsz:bsz * 2], token_type_ids[bsz * 2:]), 0)

            else:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, pos_input_ids), 0)
                attention_mask = torch.cat(
                    (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask,
                     pos_input_attention_mask),
                    0)
        else:
            combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids), 0)
            attention_mask = torch.cat(
                (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask), 0)

        outputs = self.bert(
            input_ids=combine_input,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        if self.config.geohash_open:
            mlm_scores, phrases_order_scores, tokens_order_scores, geohash_scores = self.cls(sequence_output[:bsz],
                                                                                             sequence_output[
                                                                                             bsz:bsz * 2],
                                                                                             pooled_output[
                                                                                             bsz * 2:bsz * 3])

        else:
            mlm_scores, phrases_order_scores, tokens_order_scores = self.cls(sequence_output[:bsz],
                                                                             sequence_output[bsz:bsz * 2])

        total_loss = torch.zeros([], device=self.bert.device)
        loss_fct = CrossEntropyLoss()

        masked_lm_loss = None
        order_loss = None
        cl_loss = None
        geohash_loss = None
        triplet_loss = None
        nsp_loss = None

        if mlm_labels is not None:
            masked_lm_loss = loss_fct(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss += masked_lm_loss

        if tokens_order_labels is not None and phrases_order_labels is not None:
            tokens_order_loss = loss_fct(tokens_order_scores.view(-1, self.config.tokens_order_size),
                                         tokens_order_labels.view(-1))
            phrases_order_loss = loss_fct(phrases_order_scores.view(-1, self.config.phrases_order_size),
                                          phrases_order_labels.view(-1))
            order_loss = phrases_order_loss + tokens_order_loss
            total_loss += order_loss

        if geohash_labels is not None:
            geohash_loss = torch.zeros_like(total_loss)
            for idx in range(self.config.hash_bits):
                geohash_loss += loss_fct(geohash_scores[idx].view(-1, self.config.hash_size),
                                         geohash_labels[idx].view(-1))
            geohash_loss /= self.config.hash_bits
            total_loss += geohash_loss
        cl_logits = None

        if nsp_labels is not None:
            nsp_scores = self.nsp_relationship(pooled_output[bsz * 3:])
            nsp_loss = loss_fct(nsp_scores.view(-1, 2), nsp_labels.view(-1))
            total_loss += nsp_loss

        if self.config.cl_open:
            cl_loss, cl_logits = self.cl_forward(sequence_output[bsz * 2:], attention_mask[bsz * 2:], bsz,
                                                 self.config.hard_negative)
            total_loss += cl_loss

        triplet_logits = None

        if self.config.triplet_open:
            triplet_loss, triplet_logits = self.triplet_forward(sequence_output[bsz * 2:], attention_mask[bsz * 2:],
                                                                input_ids.shape[0])
            total_loss += triplet_loss

        if masked_lm_loss is None:
            masked_lm_loss = torch.zeros_like(total_loss)

        if order_loss is None:
            order_loss = torch.zeros_like(total_loss)

        if cl_loss is None:
            cl_loss = torch.zeros_like(total_loss)

        if geohash_loss is None:
            geohash_loss = torch.zeros_like(total_loss)

        if triplet_loss is None:
            triplet_loss = torch.zeros_like(total_loss)

        if nsp_loss is None:
            nsp_loss = torch.zeros_like(total_loss)

        if not return_dict:
            output = (mlm_scores, phrases_order_scores, cl_logits, triplet_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QueryBertForPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            order_loss=order_loss,
            cl_loss=cl_loss,
            nsp_loss=nsp_loss,
            triplet_loss=triplet_loss,
            geohash_loss=geohash_loss,
            contrastive_logits=cl_logits,
            prediction_logits=mlm_scores,
            tokens_relationship_logits=tokens_order_scores,
            triplet_logits=triplet_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def cl_forward(self, input_features, attention_mask, batch_size, hard_negative=False):
        pooler_output = self.pooler(attention_mask, input_features)
        anchor_features = pooler_output[:batch_size].unsqueeze(1)
        positive_features = pooler_output[batch_size:batch_size * 2].unsqueeze(1)
        if not hard_negative:
            pooler_output = torch.cat((anchor_features, positive_features), 1)
            if self.config.pooler_type == 'cls':
                pooler_output = self.mlp(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        else:
            negative_features = pooler_output[batch_size * 2:].unsqueeze(1)
            pooler_output = torch.cat((anchor_features, positive_features, negative_features), 1)
            if self.config.pooler_type == 'cls':
                pooler_output = self.mlp(pooler_output)
            z1, z2, z3 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]

        if dist.is_initialized():
            # Gather hard negative
            if hard_negative:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        if hard_negative:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)


        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss_fct = nn.CrossEntropyLoss()

        if hard_negative:
            z3_weight = 1
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cos_sim.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)

        return loss, cos_sim

    def triplet_forward(self, input_features, attention_mask, batch_size):
        pooler_output = self.pooler(attention_mask, input_features)
        anchor_features = pooler_output[:batch_size].unsqueeze(1)
        positive_features = pooler_output[batch_size:batch_size * 2].unsqueeze(1)
        negative_features = pooler_output[batch_size * 2:batch_size * 3].unsqueeze(1)
        if self.config.pooler_type == 'cls':
            z1, z2, z3 = self.triplet_net(anchor_features, positive_features, negative_features)

        if dist.is_initialized():
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]

            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z3_list[dist.get_rank()] = z3
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
            z3 = torch.cat(z3_list, 0)

        logits, loss = self.sim(z1, z2, z3)

        return loss, logits


class QueryErnieForPreTraining(ErniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.ernie = ErnieModel(config)
        self.cls = QueryBertPreTrainingHeads(config)
        self.config = config
        # Initialize weights and apply final processing

        if config.cl_open:
            self.pooler = Pooler(config.pooler_type)
            self.sim = Similarity(config.temp)
            if config.pooler_type == 'cls':
                self.mlp = MLPLayer(config)

        elif config.triplet_open:
            self.pooler = Pooler(config.pooler_type)
            self.sim = TripletLoss(config.margin)
            if config.pooler_type == 'cls':
                self.triplet_net = TripletNet(config)

        elif config.nsp_open:
            self.nsp_relationship = nn.Linear(config.hidden_size, 2)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            mlm_input_ids: Optional[torch.Tensor] = None,
            query_input_attention_mask: Optional[torch.Tensor] = None,
            order_input_ids: Optional[torch.Tensor] = None,
            nsp_input_ids: Optional[torch.Tensor] = None,
            mlm_input_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            mlm_labels: Optional[torch.Tensor] = None,
            tokens_order_labels: Optional[torch.Tensor] = None,
            nsp_labels: Optional[torch.Tensor] = None,
            phrases_order_labels: Optional[torch.Tensor] = None,
            geohash_labels: Optional[List[torch.Tensor]] = None,
            pos_input_ids: Optional[torch.Tensor] = None,
            neg_input_ids: Optional[torch.Tensor] = None,
            pos_input_attention_mask: Optional[torch.Tensor] = None,
            neg_input_attention_mask: Optional[torch.Tensor] = None,
            order_input_attention_mask: Optional[torch.Tensor] = None,
            nsp_input_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QueryBertForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bsz = input_ids.shape[0]
        if pos_input_ids is not None and pos_input_attention_mask is not None:

            if self.config.hard_negative or self.config.triplet_open:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, pos_input_ids, neg_input_ids), 0)
                attention_mask = torch.cat(
                    (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask,
                     pos_input_attention_mask,
                     neg_input_attention_mask), 0)

            elif self.config.nsp_open:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, nsp_input_ids), 0)
                attention_mask = torch.cat((mlm_input_attention_mask, order_input_attention_mask,
                                            query_input_attention_mask, nsp_input_attention_mask), 0)
                token_type_ids = torch.cat((token_type_ids[bsz:bsz * 2], token_type_ids[bsz:bsz * 2],
                                            token_type_ids[bsz:bsz * 2], token_type_ids[bsz * 2:]), 0)

            else:
                combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids, pos_input_ids), 0)
                attention_mask = torch.cat(
                    (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask,
                     pos_input_attention_mask),
                    0)
        else:
            combine_input = torch.cat((mlm_input_ids, order_input_ids, input_ids), 0)
            attention_mask = torch.cat(
                (mlm_input_attention_mask, order_input_attention_mask, query_input_attention_mask), 0)

        outputs = self.ernie(
            input_ids=combine_input,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        if self.config.geohash_open:
            mlm_scores, phrases_order_scores, tokens_order_scores, geohash_scores = self.cls(sequence_output[:bsz],
                                                                                             sequence_output[
                                                                                             bsz:bsz * 2],
                                                                                             pooled_output[
                                                                                             bsz * 2:bsz * 3])

        else:
            mlm_scores, phrases_order_scores, tokens_order_scores = self.cls(sequence_output[:bsz],
                                                                             sequence_output[bsz:bsz * 2])

        total_loss = torch.zeros([], device=self.ernie.device)
        loss_fct = CrossEntropyLoss()

        masked_lm_loss = None
        order_loss = None
        cl_loss = None
        geohash_loss = None
        triplet_loss = None
        nsp_loss = None

        if mlm_labels is not None:
            masked_lm_loss = loss_fct(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss += masked_lm_loss

        if tokens_order_labels is not None and phrases_order_labels is not None:
            tokens_order_loss = loss_fct(tokens_order_scores.view(-1, self.config.tokens_order_size),
                                         tokens_order_labels.view(-1))
            phrases_order_loss = loss_fct(phrases_order_scores.view(-1, self.config.phrases_order_size),
                                          phrases_order_labels.view(-1))
            order_loss = phrases_order_loss + tokens_order_loss
            total_loss += order_loss

        if geohash_labels is not None:
            geohash_loss = torch.zeros_like(total_loss)
            for idx in range(self.config.hash_bits):
                geohash_loss += loss_fct(geohash_scores[idx].view(-1, self.config.hash_size),
                                         geohash_labels[idx].view(-1))
            geohash_loss /= self.config.hash_bits
            total_loss += geohash_loss

        cl_logits = None

        # if nsp_labels is not None:
        #     nsp_scores = self.nsp_relationship(pooled_output[bsz * 3:])
        #     nsp_loss = loss_fct(nsp_scores.view(-1, 2), nsp_labels.view(-1))
        #     total_loss += nsp_loss

        if self.config.cl_open:
            cl_loss, cl_logits = self.cl_forward(sequence_output[bsz * 2:], attention_mask[bsz * 2:], bsz,
                                                 self.config.hard_negative)
            total_loss += cl_loss

        triplet_logits = None

        # if self.config.triplet_open:
        #     triplet_loss, triplet_logits = self.triplet_forward(sequence_output[bsz * 2:], attention_mask[bsz * 2:],
        #                                                         input_ids.shape[0])
        #     total_loss += triplet_loss

        if masked_lm_loss is None:
            masked_lm_loss = torch.zeros_like(total_loss)

        if order_loss is None:
            order_loss = torch.zeros_like(total_loss)

        if cl_loss is None:
            cl_loss = torch.zeros_like(total_loss)

        if geohash_loss is None:
            geohash_loss = torch.zeros_like(total_loss)

        if triplet_loss is None:
            triplet_loss = torch.zeros_like(total_loss)

        if nsp_loss is None:
            nsp_loss = torch.zeros_like(total_loss)

        if not return_dict:
            output = (mlm_scores, phrases_order_scores, cl_logits, triplet_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QueryBertForPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            order_loss=order_loss,
            cl_loss=cl_loss,
            nsp_loss=nsp_loss,
            triplet_loss=triplet_loss,
            geohash_loss=geohash_loss,
            contrastive_logits=cl_logits,
            prediction_logits=mlm_scores,
            tokens_relationship_logits=tokens_order_scores,
            triplet_logits=triplet_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def cl_forward(self, input_features, attention_mask, batch_size, hard_negative=False):
        pooler_output = self.pooler(attention_mask, input_features)
        anchor_features = pooler_output[:batch_size].unsqueeze(1)
        positive_features = pooler_output[batch_size:batch_size * 2].unsqueeze(1)
        if not hard_negative:
            pooler_output = torch.cat((anchor_features, positive_features), 1)
            if self.config.pooler_type == 'cls':
                pooler_output = self.mlp(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        else:
            negative_features = pooler_output[batch_size * 2:].unsqueeze(1)
            pooler_output = torch.cat((anchor_features, positive_features, negative_features), 1)
            if self.config.pooler_type == 'cls':
                pooler_output = self.mlp(pooler_output)
            z1, z2, z3 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]

        if dist.is_initialized():
            # Gather hard negative
            if hard_negative:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        if hard_negative:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)


        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss_fct = nn.CrossEntropyLoss()

        if hard_negative:
            z3_weight = 1
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cos_sim.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)

        return loss, cos_sim

    def triplet_forward(self, input_features, attention_mask, batch_size):
        pooler_output = self.pooler(attention_mask, input_features)
        anchor_features = pooler_output[:batch_size].unsqueeze(1)
        positive_features = pooler_output[batch_size:batch_size * 2].unsqueeze(1)
        negative_features = pooler_output[batch_size * 2:batch_size * 3].unsqueeze(1)
        if self.config.pooler_type == 'cls':
            z1, z2, z3 = self.triplet_net(anchor_features, positive_features, negative_features)

        if dist.is_initialized():
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]

            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z3_list[dist.get_rank()] = z3
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
            z3 = torch.cat(z3_list, 0)

        logits, loss = self.sim(z1, z2, z3)

        return loss, logits
