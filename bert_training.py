import pickle
import torch
from torch.nn.utils import stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForPreTraining
import copy
import numpy as np
from functorch import make_functional_with_buffers
import torchopt
from torchopt.transform.scale_by_adam import ScaleByAdamState
from torchopt.base import EmptyState

from shark.shark_trainer import SharkTrainer

def get_params(named_params):
        return [i[1] for i in named_params.items()]

class BertPretrainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_model = BertForPreTraining.from_pretrained('bert-base-uncased')
        my_config = copy.deepcopy(base_model.config)
        my_config.vocab_size = 30522
        my_config.num_hidden_layers = 1
        my_config.num_attention_heads = 1
        my_config.hidden_size = 16
        my_config.hidden_dropout_prob = 0.0
        my_config.attention_probs_dropout_prob = 0.0
        self.model = BertForPreTraining(my_config)

    def forward(self, input_ids, input_mask, segment_ids, labels, next_sentence_label):
        return self.model.forward(input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels)

torch.manual_seed(0)
bert = BertPretrainer()
print(bert)
#bert = torch.load('torch_bert_1.pt')
#torch.save(bert, 'torch_bert_2.pt')
# input_ids = torch.from_numpy(np.load('input_ids.npy'))
# input_mask = torch.from_numpy(np.load('input_mask.npy'))
# masked_lm_labels = torch.from_numpy(np.load('masked_lm_labels.npy'))
# next_sentence_labels = torch.from_numpy(np.load('next_sentence_labels.npy'))
# segment_ids = torch.from_numpy(np.load('segment_ids.npy'))
input_ids = torch.randint(0, 5000, (8,128), dtype=torch.int32)
input_mask = torch.randint(0, 2, (8,128), dtype=torch.int32)
masked_lm_labels = torch.randint(1, 3000, (8,128), dtype=torch.int64)
next_sentence_labels = torch.randint(0, 2, (8,), dtype=torch.int64)
segment_ids = torch.randint(0, 2, (8,128), dtype=torch.int32)

packed_inputs = (input_ids,
                 input_mask,
                 segment_ids,
                 masked_lm_labels,
                 next_sentence_labels)
#output = bert(*packed_inputs)
func, bert_params, bert_buffers = make_functional_with_buffers(bert)
optim = torchopt.adamw(lr=4e-4)
opt_state = optim.init(bert_params)
opt_state_dict = dict(opt_state[0]._asdict())
opt_state_dict = {key: tuple(tensor_list) for key, tensor_list in opt_state_dict.items()}

def forward(params, buffers, opt_state_dict, packed_inputs):
    params_and_buffers = {**params, **buffers}
    output = stateless.functional_call(
        bert, params_and_buffers, packed_inputs, {}
    )
    bert_params_tuple = ()
    for name, param in params.items():
        bert_params_tuple += (param,)
    print('TUPLE:', bert_params_tuple)
    loss = output.loss
    print("======LOSS=====:", loss)
    grads = torch.autograd.grad(loss, bert_params_tuple)
    #print('GRADS:', grads)
    new_state = ScaleByAdamState(opt_state_dict['mu'], opt_state_dict['nu'], opt_state_dict['count'])
    empty = EmptyState()
    opt_tuple = ()
    opt_tuple += (new_state,)
    opt_tuple += (empty,)
    opt_tuple += (empty,)
    #print("params:", params)
    updates, opt_state_new = optim.update(grads, opt_tuple, params=bert_params_tuple)
    #print("updates:", type(updates))
    #print('BEFORE UPDATE:', opt_state_new)
    updates_dict = {}
    i = 0
    for key in params:
        updates_dict[key] = updates[i]
        i += 1
    #print(updates_dict)
    params = torchopt.apply_updates(params, updates_dict)
    #print("new params: ", type(params))
    params = dict(params)
    #print('AFTER UPDATE:', opt_state_new)
    #print(type(opt_state_new))
    #print('AFTER UPDATE:', params)
    return params, buffers, opt_state_new, loss

bert_train_module = SharkTrainer(bert, packed_inputs, opt_state_dict)
bert_train_module.compile(forward)
params, losses = bert_train_module.train(2)

print("done training")
