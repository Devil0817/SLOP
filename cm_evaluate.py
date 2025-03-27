#!/usr/bin/env python
# Copyright (c) 2021 EleutherAI
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

from megatron.utils import print_rank_0, setup_for_inference_or_eval,is_mp_rank_0

from megatron.text_generation_utils import (
    generate_samples_input_from_file,
    generate_samples_from_prompt,
    generate_samples_unconditional,
    generate_samples_interactive,
    generate_eval_tasks
)
import json
import time
import pandas as pd
def main():
    """
    Generate text/sample model
    """
    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    #print("##############")
    #print(model)
    #from torchsummary import summary
    #summary(model, input_size=[(1, 4096, 64768)], batch_size=1, device="cpu")

    data_dir = './eval_data/data'
    prompt_dir = './eval_data/prompts'
    prompt_idx = 0
    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing
    if neox_args.few_shot != 0:
#         neox_args.model_name += str(neox_args.few_shot) + 'shot'
        shot = str(neox_args.few_shot) + 'shot'
    else:
        shot = ''
    t1 = time.time()
    metrics = 'acc'
    for eval_task in ['wikizh', 'lambada', 'piqa', 'gsm8k', 'hellaswag', 'bbh']:
        print('eval task is:', eval_task)
        # eval_task = neox_args.eval_tasks[0]
        if eval_task in ['gsm8k']:
            end_idx = 1
        elif eval_task in ['timeclassfiy']:
            end_idx = 1
        elif eval_task in ['bbh']:
            end_idx = 1
        elif eval_task in ['hellaswag', 'piqa', 'triviaqa']:
            end_idx = 1
        elif eval_task in ['lambada']:
            end_idx = 1
            neox_args.return_logits=True
            metrics = 'ppl'
        else:
            raise ValueError(f"wrong eval task: {eval_task}")
#         print('===============maximum_tokens:', neox_args.maximum_tokens)
        for prompt_idx in range(end_idx):
            result = generate_eval_tasks(
                neox_args=neox_args,
                model=model,
                eval_task=eval_task,
                data_dir=data_dir,
                prompt_dir=prompt_dir,
                prompt_idx=prompt_idx,
                eos_token_id=neox_args.tokenizer.eod,
                maximum_tokens=neox_args.maximum_tokens,
                recompute=neox_args.recompute,
                temperature= neox_args.temperature, # 1,
                top_k= neox_args.top_k, # 5,
                top_p= neox_args.top_p, # 0.75,
            )

            t2 = time.time()
            spend_time = t2 - t1
            # 结果输出到 output_file里
            output_file = f'./eval_results/results/{eval_task}.json'

            if is_mp_rank_0():
                metric_res = get_metric(eval_task, neox_args.model_name, shot, prompt_idx, metrics, result, spend_time, output_file)


def get_metric(task, model, shot, prompt_idx, metric, result, spend_time, output_file):
    metric_res = {
        "task_name": task,
        "model":model,
        "time":spend_time,
        "metric": {}
    }
    metric_res['samples_number'] = len(result)
    cnt_eq, cnt_in = 0, 0

    if metric == 'acc':
        in_list = []
        for i,data in enumerate(result):
            
            if data["gen_text"] and data["ans_text"]:
                if str(data["gen_text"]).strip() == str(data["ans_text"]).strip():
                    cnt_eq += 1
                if str(data["ans_text"]).strip() in str(data["gen_text"]).strip():
                    cnt_in += 1
                    result[i]['acc'] = True
                else:
                    result[i]['acc'] = False
        metric_res['cnt_eq'] = cnt_eq
        metric_res['cnt_in'] = cnt_in

        if len(result) != 0:
            metric_res["metric"][metric]= cnt_in / len(result)

            
    elif metric == 'ppl':
        import numpy as np        
        ppl_ave = 0
        ppl_list = []
        for i,data in enumerate(result):
            gt = data['ans_id']
            probs = data['probs'][0][0]
            ppl = (1 / (probs + 1e-9))#.item()
            ppl_ave += ppl
            print_rank_0('=========ppl: ===========', ppl)
            result[i]['ppl'] = ppl
            
        ppl_ave = ppl_ave / len(result)
        metric_res['metric'][metric] = ppl_ave
    elif metric == 'out':
        metric_res['metric'][metric] = 'out'
    else:
        raise ValueError(f"metric: {metric} not support")

    # 写入文件并按 modelname_task 格式命名
    print(f'metric_res===================={metric_res}')
    if shot:
        model = model + '_' + shot
    with open(f'eval_results/results/{model}_{task}_p{prompt_idx}_dev.json', 'w') as f: # checkpoint_tasks_name
        f.write(json.dumps(metric_res))
    df = pd.DataFrame(result)
    df.to_csv(f'eval_results/results/{model}_{task}_p{prompt_idx}_dev.csv',index=None)
    return metric_res


if __name__ == "__main__":
    main()
