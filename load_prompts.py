import json
import pandas as pd
import jsonlines

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def load_lambada(input_file='./eval_data/data/lambada/lambada_development_plain_text.txt'):
    with open(input_file,'r', encoding='utf-8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            
            text = loop_remove(text)
            prompts = ' '.join(text)
            gt = ''
            yield prompts, gt


def load_piqa(input_file='/root/share/lm_eval/data/piqa/dev.jsonl', label_file='/root/share/lm_eval/data/piqa/dev-labels.lst',
                prompt_file='/root/share/lm_eval/llama-main/prompts/piqa_prompts.txt', prompt_idx=0, shot=0):
    prompts_mode = 'Goal: {{goal}} Which is the correct ending? - {{sol1}} - {{sol2}} Answer:'
    few_shot = ''
    label_map = {0:'a', 1:'b'}
    if shot != 0:
        examples=['Goal: Where can I buy a tennis ball  Which is the correct ending? (a) You can purchase a tennis ball at any sports store (b) You can purchase a tennis racket at any sports store Answer: a\n']
        for example in examples[:shot]:
            few_shot = few_shot + example
    
    label_list = []
    with open(label_file, 'r') as reader:
        label_list = reader.readlines()

    with jsonlines.open(input_file, "r") as reader1:
        for i, line in enumerate(reader1):
            goal = line['goal']
            sol1 = line['sol1']
            sol2 = line['sol2']
            prompt = prompts_mode.replace('{{goal}}', goal)
            prompt = prompt.replace('- {{sol1}}', '(a) ' + sol1)
            prompt = prompt.replace('- {{sol2}}', '(b) ' + sol2)
#             ans = sol1 if label_list[i] == 1 else sol2
            ans = label_map[int(label_list[i].strip())]
            prompt = prompt + '\n'
            if few_shot:
                prompt = few_shot + prompt
            yield prompt, ans
            
                
def load_gsm8k(input_file='./data/gsm8k/GSM8K.jsonl'):
    import re
    with open(input_file, "r", encoding='utf-8') as f:
        reader =f.readlines()
        for row in reader:
            line = json.loads(row.strip())
            prompt = 'Q:'
            prompt += line['question']+'\n'+'A:'
            text = line['answer']

            match = re.compile(r"#### (\-?[0-9\.\,]+)").search(line['answer'])
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
            yield prompt, match_str
            

if __name__ == "__main__":
    for i, (prompts, ans) in enumerate(load_task_test()):
        print(prompts)
        print(ans)
        if i > 20:
            break
