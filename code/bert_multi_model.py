from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union



class BertForMultiSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.input_len = 3 # Title, description, transaction description
        self.classifier = nn.Linear(config.hidden_size * self.input_len, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_list = [
            {
                'input_ids': torch.zeros_like(input_ids),
                'attention_mask': torch.zeros_like(attention_mask),
                'token_type_ids': torch.zeros_like(token_type_ids)
            },
            {
                'input_ids': torch.zeros_like(input_ids),
                'attention_mask': torch.zeros_like(attention_mask),
                'token_type_ids': torch.zeros_like(token_type_ids)
            },
            {
                'input_ids': torch.zeros_like(input_ids),
                'attention_mask': torch.zeros_like(attention_mask),
                'token_type_ids': torch.zeros_like(token_type_ids)
            },
        ]

        # Find indices for all batches where CLS (token id 101) occurs. Drop the first as it will always occur at index 0
        cls_index_list = [torch.where(row == 101)[0].tolist()[1:] for row in input_ids]
        input_tensor_variables = ['input_ids', 'attention_mask', 'token_type_ids']
        for input_tensor_variable in input_tensor_variables:
            for batch, (row, indices) in enumerate(zip(eval(input_tensor_variable), cls_index_list)):
                splits = torch.split(row, [i - j for i, j in zip(indices + [len(row)], [0] + indices)])
                for split_index, split in enumerate(splits):
                    input_list[split_index][input_tensor_variable][batch][:len(split)] += split

        pooled_outputs = list()
        for i in range(0, self.input_len):
            outputs = self.bert(
                input_list[i]['input_ids'][:,:self.config.max_position_embeddings],
                attention_mask=input_list[i]['attention_mask'][:,:self.config.max_position_embeddings],
                token_type_ids=input_list[i]['token_type_ids'][:,:self.config.max_position_embeddings],
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_outputs.append(outputs.pooler_output)
        pooled_output = torch.stack(pooled_outputs, dim=1)
        batch_size = outputs.last_hidden_state.shape[0]
        pooled_output = pooled_output.view(1, -1).reshape(batch_size, self.config.hidden_size * self.input_len)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )