import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "meta-llama/Llama-2-7b-chat-hf"
hf_auth_token = "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"
kwargs = {"use_auth_token": hf_auth_token}
llama_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                **kwargs,
)
llama_tokenizer.padding_side = "left"
llama_tokenizer.pad_token_id = 0
kwargs["torch_dtype"] = torch.float32

llama_model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
)


from transformers.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)
import copy
from typing import Optional, List, Union, Callable, Dict, Any
from transformers.generation.streamers import BaseStreamer
from torch import nn


def prepare_inputs_for_generation(
    input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = (torch.minimum(position_ids[:, -1], torch.tensor([399]))).unsqueeze(-1)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    if past_key_values is not None:
        attention_mask = attention_mask[:, 1:] 
        pkv = []
        for tup in past_key_values:
            pkv.append((tup[0][:, :, 1:, :], tup[1][:, :, 1:, :]))
        past_key_values = tuple(pkv)

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

from transformers.utils import ModelOutput


def _update_model_kwargs_for_generation(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    model_kwargs["past_key_values"] = outputs.past_key_values

    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )
    return model_kwargs


def sample(
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)

    scores = None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    iter_counter = 0
    import time
    start = time.time()
    while True:
        # prepare model inputs
        model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

        model_kwargs["attention_mask"] = model_inputs["attention_mask"]
        if "past_key_values" in model_kwargs.keys():
            model_kwargs["past_key_values"] = model_inputs["past_key_values"]
        
        # forward pass to get next token
        outputs = llama_model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        _detok = llama_tokenizer.decode(next_tokens.numpy(), skip_special_tokens=False)
        if _detok == "<0x0A>":
            print("\n", end="", flush=True)
        else:
            print(f" {_detok}", end=" ", flush=True)

        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

        iter_counter = iter_counter + 1

    if streamer is not None:
        streamer.end()

    end = time.time()

    print(
        "\n\nTime taken is {:.2f} seconds/token\n".format(
            (end - start) / iter_counter
        )
    )
    return input_ids


def generate(
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):

    synced_gpus = False

    if generation_config is None:
        generation_config = llama_model.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()

    inputs_tensor, model_input_name, model_kwargs = llama_model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = False #generation_config.output_attentions
    model_kwargs["output_hidden_states"] = False #generation_config.output_hidden_states
    model_kwargs["use_cache"] = True #generation_config.use_cache

    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids.shape[-1]
    generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    # 8. prepare distribution pre_processing samplers
    logits_processor = llama_model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 9. prepare stopping criteria
    stopping_criteria = llama_model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    # 11. prepare logits warper
    logits_warper = llama_model._get_logits_warper(generation_config)

    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = llama_model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=False,
        **model_kwargs,
    )

    # 13. run sample
    return sample(
        input_ids,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )


while True:
    prompt = input("User: ")
    prompt_template=f'''System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    User: {prompt}
    Assistant:
    '''
    inputs = llama_tokenizer(prompt_template, padding="max_length", max_length=400, add_special_tokens=False, return_tensors="pt")
    inputs = {'input_ids' : inputs['input_ids'], 'attention_mask' : inputs['attention_mask']}
    output = generate(**inputs, temperature=0.7, max_new_tokens=512, top_p=0.95, repetition_penalty=1.15)
    print("\nResult: ", llama_tokenizer.decode(output[0]))
