# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for generating text."""
import copy
import json
import os
import time
from typing import List, Union
import pdb
import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0
# from lm_eval_tasks.load_prompts import load_CMRC2018
import sys
sys.path.append("..")
# from load_prompts import load_squad, load_tnews, load_CMRC2018, load_afqmc, load_sst2, load_wmt, load_drcd, load_lambada, load_wikitext2, load_wikizh, load_ceval, load_cmmlu, load_mmlu, load_gsm8k, load_agieval
from load_prompts import *
import numpy as np
def get_batch(neox_args, context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.

    neox_args: NeoXArgs.
    context_tokens: torch tensor with dimensions [batch, context_size]

    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    return tokens, attention_mask, position_ids


def pad_batch(context_tokens: List[List[int]], pad_id: int, pad_len: int):
    """
    pads context lengths in context_tokens with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.

    context_tokens: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, context length to be padded; all batch items will be padded to the same length

    returns: tuple of padded context tokens and a list of unpadded token count
    """

    context_lengths = []
    for tokens in context_tokens:
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
        elif context_length > pad_len:
            raise ValueError("context_length is bigger than to be padded length")
        context_lengths.append(context_length)
    return context_tokens, context_lengths


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filters the logits using top_k / top_p, filling any filtered vocab items with filter_value (defaults to -inf).

    This function has been mostly taken from huggingface conversational ai code at
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    logits: torch.Tensor -> logits of megatron model.
    top_k: integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p: float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    returns: (filtered) logits"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_model(model, model_inputs, is_pipe_parallel=False) -> torch.Tensor:
    """
    Runs model.forward(model_inputs)

    We need to create a wrapper for this function because deepspeed pipe parallel modules operate differently to normal models.

    model: a Megatron model.
    model_inputs: tuple containing model args

    returns: torch.Tensor containing the logits of the model
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model differently to a normal model
    if not is_pipe_parallel:
        return model.module(model_inputs)
    else:
        # we need to format inputs this way because:
        # a) deepspeed pipeline only accepts iterables
        # b) deepspeed pipeline *requires* that you pass in labels for the loss, it's not easy to get around this
        # so we wrap the inputs in an iterable, and pad them (because internally, we get labels as inputs[:, 1:] and inputs as inputs[:, :-1])
#         print_rank_0("======model_inputs_forward_model======", F.pad(model_inputs[0], pad=(0, 1)))
        model_inputs = iter([{"text": F.pad(model_inputs[0], pad=(0, 1))}])

        # set num microbatches to 1 at inference time
        micro_batches_before = model.micro_batches
        model.micro_batches = 1

        # deepspeed sends metadata across pipeline stages only once in the first step, then assumes it will stay
        # constant. In inference, the metadata of the tensors being sent across pipe stages may change, so we need to set
        # these two flags in order for deepspeed to send the metadata every step, otherwise torch.distributed hangs
        # silently. Fun stuff.
        model.first_output_send = True
        model.pipe_recv_buf = None

        loss, logits = model.eval_batch(model_inputs, return_logits=True)
        model.micro_batches = micro_batches_before
        return logits


def broadcast_terminate_signal(terminate_runs: int):
    """Send signal to all workers to terminate if we've finished the process"""
    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(
        terminate_runs_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    return terminate_runs_tensor[0].item()


def stop_tokens_in_completion(stop_tokens, context_tokens, batch_index, current_index):
    if stop_tokens is None:
        return False
    res = []
    for token_group in stop_tokens:
        context = context_tokens[batch_index, : current_index + 1]
        context = context[-len(token_group) :]
        if context.shape[0] == token_group.shape[0]:
            res.append(all(token_group == context))
        else:
            res.append(False)
    return any(res)


def stream_tokens(
    neox_args,
    model,
    gt: int,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    iterator producing text completions

    neox_args: NeoXArgs.
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids
    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.
    maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards.
                    longer batch items get less generated tokens.
    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
    yields: (
                tokens (completions from model),
                token_generation_start_index (token index per batch item for the first generated token),
                token_generation_end_index (token index per batch item for the last generated token),
                logits (logits which are so far computed, zeros otherwise),
                is_done (flag for each bach item indicating whether an eod token was generated)
            )

            * each iteration adds a generated token to the context_tokens
            * output contains both context_tokens from input and generated tokens
            * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
    """

    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        if len(stop_tokens) > 0 and type(stop_tokens[0]) is not list:
            stop_tokens = [stop_tokens]
        for i in range(0, len(stop_tokens)):
            stop_tokens[i] = torch.cuda.LongTensor(stop_tokens[i])

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
#     print('context_lengths:', context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)
    
    # set variables
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    maximum_tokens = maximum_tokens or (
        neox_args.seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )
#     pdb.set_trace()
    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)
        generation_logits = (
            torch.empty(maximum_tokens, neox_args.padded_vocab_size).float().cuda()
        )
#         neox_args.is_pipe_parallel=False
#         print_rank_0("====is_pipe_parallel====",neox_args.is_pipe_parallel)
        while token_index_to_generate <= last_token_index_to_generate:
#             recompute=True
#             print_rank_0("======token_index_to_generate=======", token_index_to_generate)
            if recompute:  # recompute all tokens
#                 print_rank_0("======recompute=====",recompute)
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
#                 if token_index_to_generate == first_token_index_to_generate:
# #                     print_rank_0("======first_token_index_to_generate======", first_token_index_to_generate)
#                     tokens_to_use = context_tokens[:, :token_index_to_generate]
# #                     print_rank_0("=======tokens_to_use======", tokens_to_use)
#                     positions_to_use = position_ids[:, :token_index_to_generate]
#                 else:
# #                     print_rank_0('=======context_tokens========', context_tokens.cpu().tolist())
#                     tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
#                         batch_size, -1
#                     )
# #                     print_rank_0("=======tokens_to_use======", tokens_to_use)
#                     positions_to_use = position_ids[
#                         :, token_index_to_generate - 1
#                     ].view(batch_size, -1)
                model.module.clear_cache()
                tokens_to_use = context_tokens[:, :token_index_to_generate]
                positions_to_use = position_ids[:, :token_index_to_generate]
                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )
#                 print_rank_0('=======model_inputs======', model_inputs)
#                 pdb.set_trace()
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
#                 print('=====logits shape=====', logits.shape)
#                 print(f'========logits {logits}=========')
#                 if logits is not None:  # if pipe parallel, not all ranks return logits

            if logits is not None:
                generated_token_logits = (
                    logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]
#                 temperature, top_k, top_p = 0.0, 0, 0.0 # 0.8, 5, 0.75 # 
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
#                     if generated_tokens[0] >64645 or generated_tokens[0]==0:
#                         print_rank_0('=======generated_tokens========', generated_tokens)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    print(next_token_log_probs.topk(top_k))
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

            if neox_args.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                
                generated_token_logits = (
                    generated_token_logits
                    if logits is not None
                    else torch.zeros((batch_size, neox_args.padded_vocab_size), dtype=torch.float16).cuda()
                )

                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
#                 print(f'======generated_tokens before braodcast={generated_tokens}=======')
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )
#                 print(f'======generated_tokens after broadcast={generated_tokens},type={type(generated_tokens)}=======')
            prob = None
            if neox_args.return_logits:
                probs = torch.softmax(generated_token_logits, -1)
                gt = torch.tensor([gt],dtype=int).cuda()
#                 print('======gather gt=====',gt)
                prob = torch.gather(probs, 1, gt)
#                 print('======gather prob=====', prob)
                if neox_args.is_pipe_parallel:
                    prob = (
                        torch.tensor(prob,dtype=torch.float16)
                        if logits is not None
                        else torch.zeros_like(prob).cuda()
                        )

                    torch.distributed.broadcast(
                        tensor=prob,
                        src=src_rank,
                        group=mpu.get_pipe_parallel_group(),
                        )    
            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started
#             print_rank_0('=======context_tokens========', context_tokens.cpu().tolist())
#             print_rank_0('=======generated_tokens========', generated_tokens.cpu().tolist())
            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )
#             print_rank_0('=======context_tokens========', context_tokens.cpu().tolist())
            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1
#             if generated_tokens[0] >64645 or generated_tokens[0]==0:
#                 print_rank_0('=======context_tokens========', context_tokens)
            yield context_tokens, token_generation_start_index, token_generation_end_index, generated_token_logits, prob, state_is_done.bool()
            if torch.all(state_is_done):
                break


def generate_samples_from_prompt(
    neox_args,
    model,
    text: Union[List[str], str],
    ans,
    eos_token_id: int = None,
    maximum_tokens: int = 8,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
    metrics='acc'
):
    """
    Generates samples from raw text and returns them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds

    """
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    start_time = time.time()
    # type check
    assert any(
        [isinstance(text, str), isinstance(text, list)]
    ), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]

    input_count = len(text)
    input_pos = 0

    # generate completions
    generated_texts = []
    while True:
        model.module.clear_cache()  # clear kv cache between batches
#         start_time = time.time()
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if input_pos == input_count:
            terminate_runs = 1
        else:
            raw_text = text[input_pos]
            input_pos += 1

            if raw_text == "":
                context_tokens = [eos_token_id]
            else:
                context_tokens = neox_args.tokenizer.tokenize(raw_text)
                ans = neox_args.tokenizer.tokenize(raw_text)
            context_length = len(context_tokens)

            if context_length >= (neox_args.seq_length // 2):
                print_rank_0(
                    "\nWarning! Context length",
                    context_length,
                    "\nPlease give smaller context (e.g. half of the "
                    "max sequence length)!",
                )
        if not is_mp_rank_0():
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts
#         if metrics == 'ppl':
#             ans = context_tokens[-1]
#             context_tokens = context_tokens[:-1]
#             maximum_tokens=1
#             print(ans, context_tokens)
        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            batch_prob,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            gt=ans, # [ans]
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pass  # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = (
            batch_token_generation_start_index.cpu().numpy().tolist()
        )
        batch_token_generation_end_index = (
            batch_token_generation_end_index.cpu().numpy().tolist()
        )
        batch_is_done = is_done.cpu().numpy().tolist()

        for tokens, start_index, end_index, is_done in zip(
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_is_done,

        ):

            if end_index >= start_index:
                generated_tokens = tokens[start_index : end_index + 1]
                try:
#                   
                    generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                    message = None
                except KeyError:
                    generated_text = None
                    message = "WARNING: generated token which doesn't exist."
            else:
                generated_text = None
                generated_tokens = []
                # this will happen if the first generated token is a stop token or eos token
                message = "WARNING: text generation did not start; try different batching or adjust parameters"
            if is_mp_rank_0():
                for single_token in generated_tokens:
                    single_decode = neox_args.tokenizer.detokenize([single_token])
                    print('==========',single_token, single_decode)
                data = {
                    "context": raw_text,
                    "text": generated_text,
                    "length": len(generated_tokens),
                    "finished": is_done,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),
                }

                if neox_args.return_logits:

                    data["probs"] = batch_prob[0]
                generated_texts.append(data)
#     print('ans:')
#     print(generated_texts)
    return generated_texts


def generate_samples_input_from_file(
    neox_args,
    model,
    input_file,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 8,
    prompt_end: str = "\n",
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    few_shot=0
):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file

    neox_args: NeoXArgs.
    model: a Megatron model

    input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
    output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated
    prompt_end: end of a single input prompt. Defaults to newline character '\n'. Other prompt-end sequences may be useful when generating indent-aware completions (e.g. code)

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0


    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """
    # Read the sample file
    print_rank_0(
        "generate_samples_input_from_file() loading input from {}".format(input_file)
    )
#     prompts=[]
#     with open(input_file, "r", encoding="utf-8") as f:
#         prompts = f.read()
#         prompts = prompts.split(prompt_end)
    ans_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = f.readlines()
    metrics=''
#     prompts=[]
#     with open(input_file, 'r') as f:
#         lines=f.readlines()
#         for line in lines:
#             tmp = json.loads(line.strip())
#             print(tmp)
#             prompts.append(tmp['request'])

#     with open(input_file, "r", encoding='utf-8') as f:
#         lines = f.readline()
#         lines = json.loads(lines) 
#         for k in lines:
#             prompt = lines[k]['origin_prompt']
#             prompts.append(prompt)
    # load ceval for ppl:
#     result = load_ceval_ppl_single(input_file, few_shot)
#     prompts = []
#     ans_list = []
#     metrics= 'ppl'
#     for i, (raw_text, ans) in enumerate(result):
#         prompts.append(raw_text)
#         if metrics == 'ppl':
#             # ans_list.append(raw_text[-1])

#             neox_args.return_logits=True
        
#         ans_list.append(ans)

#     result = load_ceval_single(input_file, few_shot)
#     prompts = []
#     ans_list = []
#     metrics='acc'
#     for i, (raw_text, ans) in enumerate(result):
#         prompts.append(raw_text)

#     print(prompts)
    
    filename = input_file.split('/')[-1]

    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0(
        "generate_samples_input_from_file() prompts loaded: {}".format(len(prompts))
    )
    print_rank_0(prompts[0])
    if is_mp_rank_0():
        if output_file is None:
            output_file = str(input_file) + ".output.jsonl"
            print_rank_0(
                "generate_samples_input_from_file() setting default output file to {}".format(
                    output_file
                )
            )

    print_rank_0("generate_samples_input_from_file() generating...")
    outputs = []
    for i, prompt in enumerate(prompts):
        generated_texts = generate_samples_from_prompt(
            neox_args=neox_args,
            model=model,
            text=prompt,
            ans=ans_list,
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            metrics=metrics
        )
#         print_rank_0(generated_texts)

        if is_mp_rank_0():
            item = generated_texts[0]
            if ans_list:
                probs = item['probs'].cpu().numpy()[0]
                ppl = (1 / (probs + 1e-9))
                outputs.append([item['context'],item['text'], ppl, ans_list[i]])
            else:
                outputs.append([item['context'],item['text']])
#     if is_mp_rank_0():
#         with open(output_file, "w") as f_out:
#             for item in generated_texts:
#                 f_out.write(json.dumps(item) + "\n")
#     print_rank_0("generate_samples_input_from_file() done")

#     if is_mp_rank_0():
#         with open(output_file, "w") as f_out:
#             for item in generated_texts:
#                 output={"origin_prompt":item['context'], "prediction":item['text']}
#                 f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
    if is_mp_rank_0():
        if ans_list:
            df = pd.DataFrame(outputs, columns=['context','gen_text', 'ppl', 'ans'])
        else:
            df = pd.DataFrame(outputs, columns=['context','gen_text'])
        df.to_csv(output_file, index=None)
    return generated_texts


def generate_samples_unconditional(
    neox_args,
    model,
    number_of_samples: int = 10,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    number_of_samples (default 10): number of unconditional samples to be generated

    output_file: file where generation results are to be stored in jsonl format. no file will be stored if omitted

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated
    prompt_end: end of a single input prompt. Defaults to newline character '\n'. Other prompt-end sequences may be useful when generating indent-aware completions (e.g. code). The interactive mode will reroll the user-input request until the stop-char is met

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    print_rank_0("generate_samples_unconditional() generating...")
    assert number_of_samples > 0, "number_of_samples must be > 0"
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=["" for _ in range(number_of_samples)],
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if is_mp_rank_0():
        if output_file is not None:
            with open(output_file, "w") as f_out:
                for item in generated_texts:
                    f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_unconditional() done")
    return generated_texts


def generate_samples_interactive(
    neox_args,
    model,
    maximum_tokens: int = 64,
    prompt_end: str = "\n",
    eos_token_id: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    maximum_tokens: maximum number of tokens to be generated
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    while True:
        model.module.clear_cache()  # clear kv cache between batches
        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        terminate_runs = 0

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            os.system("clear")
            raw_text = ""
            while True:
                current_input = input("Context prompt >>> ")
                if (
                    prompt_end == "\n"
                ):  # we need to handle '\n' case as 'input' strips it and leads to lines being squashed
                    raw_text += current_input
                    break
                if prompt_end in current_input:
                    raw_text += current_input.split(prompt_end)[0]
                    break
                raw_text += (
                    current_input + "\n"
                )  # re-add newline since we stripped it on input
            context_tokens = neox_args.tokenizer.tokenize(raw_text)
            if len(context_tokens) == 0:
                context_tokens = [neox_args.tokenizer.eod]
            context_length = len(context_tokens)
            if context_length >= (neox_args.seq_length - 1):
                print_rank_0(
                    "\nContext length"
                    + str(context_length)
                    + "\nReached max sequence length!"
                )
                terminate_runs = 1
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return
        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            if mpu.get_model_parallel_rank() == 0:
                generated_tokens = (
                    batch_context_tokens[0]
                    .cpu()
                    .numpy()
                    .tolist()[
                        batch_token_generation_start_index[0]
                        .item() : batch_token_generation_end_index[0]
                        .item()
                        + 1
                    ]
                )
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                print_rank_0("Generated Text: " + generated_text)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            _ = input("\n<press enter to continue>")



def generate_eval_tasks(
    neox_args,
    model,
    eval_task,
    data_dir,
    prompt_dir,
    prompt_idx=0,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    Generates samples from raw text and returns them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    data_dir: path to store data
    prompt_dir: path to store prompt files
    prompt_id: indicate the index of prompt mode in one prompt file will be used

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds

    """
#     import pdb
#     pdb.set_trace()
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    print('vocab size',len(neox_args.tokenizer.vocab.keys()))
    # type check
    # assert any(
    #     [isinstance(text, str), isinstance(text, list)]
    # ), "Text should be in string or list form"
    # if isinstance(text, str):
    #     text = [text]

    # input_count = len(text)
    # input_pos = 0

    # generate completions
    generated_texts = []
    cnt_eq = 0
    cnt_in = 0
    terminate_runs = 0
    metric = 'acc'
    if eval_task == 'squad':
        data_dir = os.path.join(data_dir, 'SQuAD/dev-v1.1.json')
        prompt_dir = os.path.join(prompt_dir, 'squadv2_prompts.txt')
        result = load_squad(data_dir, prompt_dir, prompt_idx)
        maximum_tokens = 256
    elif eval_task == 'cmrc2018':
        data_dir = os.path.join(data_dir, 'CLUEdatasets/cmrc/dev.json')
        prompt_dir = os.path.join(prompt_dir, 'cmrc_prompts.txt')
        result = load_CMRC2018(data_dir, prompt_dir,prompt_idx)
        maximum_tokens = 64
    elif eval_task == 'tnews':
        data_dir = os.path.join(data_dir, 'CLUEdatasets/tnews/dev.json')
        prompt_dir = os.path.join(prompt_dir, 'tnews_prompts.txt')
        result = load_tnews(data_dir, prompt_dir, prompt_idx, neox_args.few_shot)
        maximum_tokens = 5
    elif eval_task == 'afqmc':
        data_dir = os.path.join(data_dir, 'CLUEdatasets/afqmc/dev.json')
        prompt_dir = os.path.join(prompt_dir, 'afqmc_prompts.txt')
        result = load_afqmc(data_dir, prompt_dir, prompt_idx, neox_args.few_shot)
        maximum_tokens = 2
    elif eval_task == 'sst-2':
        data_dir = os.path.join(data_dir, 'SST-2/dev.tsv')
        result = load_sst2(data_dir)
        maximum_tokens = 5
    elif eval_task == 'drcd':
        data_dir = os.path.join(data_dir, 'CLUEdatasets/drcd/dev.json')
        prompt_dir = os.path.join(prompt_dir, 'drcd_prompts.txt')
        result = load_drcd(data_dir, prompt_dir, prompt_idx)
        maximum_tokens = 64
    elif eval_task == 'wmt':
        en_dir = os.path.join(data_dir, 'WMT18/dev/newsdev2017.tc.en.json.json')
        zh_dir = os.path.join(data_dir, 'WMT18/dev/newsdev2017.tc.zh.json.json')
        result = load_wmt(en_dir, zh_dir)
        maximum_tokens = 128
    elif eval_task == 'lambada':
        data_dir = os.path.join(data_dir, 'lambada/lambada_development_plain_text.txt')
        result = load_lambada(data_dir)
        maximum_tokens = 1
        metric = 'ppl'
    elif eval_task == 'wikitext2':
        data_dir = os.path.join(data_dir, 'wikitext-2/wiki.valid.tokens')
        result = load_wikitext2(data_dir)
        maximum_tokens = 1
        metric = 'ppl'
    elif eval_task == 'wikizh':
        data_dir = os.path.join(data_dir, 'wiki-zh/wiki_zh_dev.txt')
        result = load_wikizh(data_dir)
        maximum_tokens = 1
        metric = 'ppl'
    elif eval_task == 'ceval':
#         data_dir = os.path.join(data_dir, 'ceval-exam/test')
        data_dir = os.path.join(data_dir, 'ceval-exam/val')
        result = load_ceval(data_dir, neox_args.few_shot)
        maximum_tokens = 5
        #maximum_tokens = 10
    elif eval_task == 'cmmlu':
        data_dir = os.path.join(data_dir, 'CMMLU/test')
        result = load_cmmlu(data_dir, neox_args.few_shot)
        maximum_tokens = 5
    elif eval_task == 'agieval':
        data_dir = os.path.join(data_dir, 'AGIEval')
        result = load_agieval(data_dir, neox_args.few_shot)
        maximum_tokens = 10
    elif eval_task == 'agievalall':
        data_dir = os.path.join(data_dir, 'AGIEval-all')
        result = load_agievalall(data_dir)
        maximum_tokens = 10
    elif eval_task == 'mmlu':
        # data_dir = os.path.join(data_dir, 'MMLU/test')
        data_dir = os.path.join(data_dir, 'MMLU/val')
        result = load_mmlu(data_dir)
        maximum_tokens = 5
    elif eval_task == 'gsm8k':
        data_dir = os.path.join(data_dir, 'GSM8K.jsonl')
        result = load_gsm8k(data_dir)
        maximum_tokens = 15
    elif eval_task == 'gsm8kzh':
        data_dir = os.path.join(data_dir, 'gsm8k_tr.jsonl')
        result = load_gsm8kzh(data_dir)
        maximum_tokens = 256
    elif eval_task == 'bbh':
        data_dir = os.path.join(data_dir, 'BBH')
        result = load_bbh(data_dir)
        maximum_tokens = 5   
    elif eval_task == 'piqa':
        input_dir = os.path.join(data_dir, 'piqa/dev.jsonl')
        label_dir = os.path.join(data_dir, 'piqa/dev-labels.lst')
        result = load_piqa(input_dir, label_dir, shot=neox_args.few_shot)
        maximum_tokens = 10 # 128
    elif eval_task == 'hellaswag':
        data_dir = os.path.join(data_dir, 'hellaswag/hellaswag_val.jsonl')
        result = load_hella(data_dir, shot=neox_args.few_shot)
        maximum_tokens = 5 # 32
    elif eval_task == 'humaneval':
        data_dir = os.path.join(data_dir, 'HumanEval/HumanEval-origin.jsonl')
        result = load_humaneval(data_dir)
        maximum_tokens = 256
        metric = 'bleu'
    elif eval_task == 'timeclassfiy':
        input_dir = os.path.join(data_dir, 'timeclassify/dev.jsonl')
        result = load_time(input_dir)
        maximum_tokens = 1000
    elif eval_task == 'cmrc_context':
        data_dir = os.path.join(data_dir, 'cmrc2018_train_context.txt')
        result = load_cmrc_context(data_dir)
        maximum_tokens = 1
        metric = 'ppl'
    elif eval_task == 'cmrc_prompt':
        data_dir = os.path.join(data_dir, 'cmrc_prompt_dev.csv')
        result = load_cmrc_prompt(data_dir)
        maximum_tokens = 64
    elif eval_task == 'task1':
#         data_dir = os.path.join(data_dir, 'task_xlsx/task_result_details_f9ccbbe3-5b7c.csv')
        result = load_task_test()
        maximum_tokens = 128
#         metric='ppl'
    elif eval_task == 'gaokaobench':
        data_dir = os.path.join('/zhangpai31a/zhengy/project/test/gaokao_process/data', 'GaokaoBench_2010-2022_Political_Science_MCQs_0.json')
        result = load_gaokaobench(data_dir)
        maximum_tokens = 128
    elif eval_task == 'other':
        data_dir = os.path.join(data_dir, 'other.txt')
        result = load_file(data_dir)
        maximum_tokens = 64
    elif eval_task == 'ceval_ppl':
        data_dir = os.path.join(data_dir, 'ceval-exam/test')
        result = load_ceval_ppl(data_dir)
        maximum_tokens = 1
        metric = 'ppl'
    else:
        raise ValueError(
            f"load eval task: {eval_task} is not support"
        )
    #print_rank_0(f'eval_task===={eval_task}, len={len(list(result))}')
    for i, (raw_text, ans) in enumerate(result):
#         if i > 0:
#             break
        model.module.clear_cache()  # clear kv cache between batches

        start_time = time.time()
        # Tokenize text, and check whether we should terminate process
#         print('======raw_text====', raw_text)
        if raw_text == "":
            context_tokens = [eos_token_id]
        else:
            context_tokens = neox_args.tokenizer.tokenize(raw_text)
        context_length = len(context_tokens)
#         print('=======context_tokens_1=======', context_tokens)
        if metric == 'ppl':
#             print_rank_0('====shape=====', len(context_tokens))
            ans = context_tokens[-1]
            context_tokens = context_tokens[:-1]
#             print_rank_0('======raw_text====', raw_text)
#             print_rank_0('=======ans, context======', ans, context_tokens)
        if context_length >= (neox_args.seq_length // 2):
            print_rank_0(
                "\nWarning! Context length",
                context_length,
                "\nPlease give smaller context (e.g. half of the "
                "max sequence length)!",
            )
        if not is_mp_rank_0():
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts
        # Added by Chong
        # if i%50 == 0:
        #     print_rank_0(f'i===={i}, raw_text={raw_text}, ans={ans}')

        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            batch_prob,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            gt=[ans],
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pass  # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = (
            batch_token_generation_start_index.cpu().numpy().tolist()
        )
        batch_token_generation_end_index = (
            batch_token_generation_end_index.cpu().numpy().tolist()
        )
        batch_is_done = is_done.cpu().numpy().tolist()

        for tokens, start_index, end_index, is_done in zip(
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_is_done,
        ):

            if end_index >= start_index:
                generated_tokens = tokens[start_index : end_index + 1]
                
                try:
#                     if np.isnan(generated_tokens):
#                         print("===type: float:====",generated_tokens)
#                     for tok in reversed(generated_tokens):
#                         if tok >= 64645:#127969:
#                             generated_tokens.remove(tok)
#                             print("out of range,token= ", tok)
#                         if tok == 0:
#                             print("out of range,token= ", tok)

                    generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                    message = None
                except KeyError:
                    generated_text = None
                    message = "WARNING: generated token which doesn't exist."
            else:
                generated_text = None
                generated_tokens = []
                # this will happen if the first generated token is a stop token or eos token
                message = "WARNING: text generation did not start; try different batching or adjust parameters"
            if is_mp_rank_0():
                # print('======generated_tokens======',generated_tokens)
                for single_token in generated_tokens:
                    single_decode = neox_args.tokenizer.detokenize([single_token])
                    # print('==========',single_token, single_decode)
                if metric == 'ppl':
                    ans_decode = neox_args.tokenizer.detokenize(ans)
                    context = neox_args.tokenizer.detokenize(context_tokens)
                else:
                    ans_decode = ans
                    context = raw_text
                data = {
                    "context": context,
                    "ans_text": ans_decode,
                    "ans_id": ans,
                    "gen_text": generated_text,
                    "gen_id": generated_tokens,
                    "length": len(generated_tokens),
                    "finished": is_done,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),                    
                }
                if eval_task == 'humaneval' and generated_text is not None:
                    # def_index = generated_text.find('\ndef')
                    # if def_index >= 0:
                    #     generated_text = generated_text[0:def_index]
                    generated_text = raw_text + generated_text
                # if i%1 == 0:
                #     print_rank_0('=======generated_text======',generated_text)
#                 print_rank_0(data)
                if neox_args.return_logits:
                    
#                     print(f'=======batch_generated_token_logits={batch_generated_token_logits}=======')
#                     print(f'=======max batch_generated_token_logits of rank {mpu.get_model_parallel_rank()}={torch.max(batch_generated_token_logits, -1)}======' )
#                     probs = torch.softmax(batch_generated_token_logits, dim=-1)
#                     print(f'=======data probs(cuda) of rank {mpu.get_model_parallel_rank()}={probs}======')
#                     print(f'=======max probs(cuda) of rank {mpu.get_model_parallel_rank()}={torch.max(probs, -1)}======' )
                    probs = batch_prob
                    data["probs"] = probs.cpu().numpy()
#                     print(f'=======data probs(cpu) of rank {mpu.get_model_parallel_rank()}={data["prob"]}=======')
#                     print(f'=======probs(cpu) shape of rank {mpu.get_model_parallel_rank()}={data["prob"].shape}=======')
                    
                generated_texts.append(data)

        #if (i >= 999 and (eval_task not in ['cmmlu', 'mmlu', 'ceval', 'agieval', 'agievalall', 'bbh', 'gsm8k', 'humaneval'])): #or (i>=1 and eval_task == 'humaneval'):
        if (i >= 999) and (eval_task != 'task1') and (eval_task != 'ceval_ppl') and (eval_task != 'ceval'): #or (i>=1 and eval_task == 'humaneval'):
            break
            
#     print_rank_0(f'generated_texts===={generated_texts}')
    return generated_texts


