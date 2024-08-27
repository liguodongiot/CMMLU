import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval, extract_choice
from tqdm import tqdm
import requests
import json
from time import sleep
from transformers import AutoTokenizer, AutoModel
import re


url = "http://10.112.2.145:9009/v1/chat/completions"

# tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen1.5-7B-Chat", trust_remote_code=True)

def get_response(inputs):
    timeout_counter = 0
    completion = None
    while completion is None and timeout_counter<=30:
        try:
            messages = [
                {"role": "user", "content": inputs}
                ]
            payload = {
                "model": "qwen1.5",
                "messages": messages,
                "max_tokens": 256,
                "top_p": 0.95,
                "seed": 100,
                "temperature": 0.8,
                "stream": False
            }
        
            headers = {"content-type": "application/json"}
            response = requests.request("POST", url, json=payload, headers=headers)
            # print(response.text)
            response_json = json.loads(response.text)
            response_str = response_json['choices'][0]['message']['content']
            return response_str
        except Exception as msg:
            if "timeout=600" in str(msg):
                timeout_counter+=1
            print(msg)
            sleep(5)
            continue
            print("Some error occured when getting gpt output.")



def eval(tokenizer,
    subject, dev_df, test_df, 
    num_few_shot, max_length, cot, **kwargs):
    
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0])):
        
        # 封装请求提示，不包含答案 
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        
        prompt = gen_prompt(dev_df, 
                            subject, 
                            prompt_end, 
                            num_few_shot, 
                            tokenizer, 
                            max_length, 
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        print('\n---------\n', i, prompt)

        pred = get_response(prompt)
        
        print("正确答案：", label)
        print("实际答案：", pred)
        
        # ext_answer = extract_ans(pred)
        extract_answer = extract_choice(pred)
        print("抽取的实际答案：", extract_answer)
        
        # if pred and pred[0] in choices:
        #     cors.append(pred[0] == label)
        
        cors.append(extract_answer == label)
        all_preds.append(pred.replace("\n", "") if pred is not None else "")

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None



def extract_ans(response_str):
        pattern=[
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"选([A-D])",
            r"选项([A-D])",
        ]
        
        ans_list=[]
        
        # if response_str[0] in ["A",'B','C','D']:
        if response_str and response_str[0] in choices:
            ans_list.append(response_str[0])
        
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--data_dir", "-d", type=str, default="../data")
    parser.add_argument("--save_dir", "-s", type=str, default="../results/GPT4")
    parser.add_argument("--num_few_shot", "-n", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen1.5-7B-Chat", trust_remote_code=True)

    run_eval(None, tokenizer, eval, args)

