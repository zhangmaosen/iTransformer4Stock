import os
os.environ['VLLM_USE_MODELSCOPE'] = "True"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,0" #需要在import torch前设置

import torch
torch.cuda.is_available()
torch.cuda.get_device_capability()
from vllm import LLM
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
#model_name = "deepseek-ai/deepseek-llm-7b-chat"
model_name = "/data/models/qwen-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name, trust_remote_code=True,dtype=torch.float16,tensor_parallel_size=2)

from vllm import LLM, SamplingParams
import pandas as pd
import json
data_path = "/data/stock_datasets/stock_news.jsonl"
df_news = pd.read_json(data_path,lines=True)
from transformers import AutoModel
from numpy.linalg import norm

model = AutoModel.from_pretrained('/data/models/jina_emb/', trust_remote_code=True) 
model.to('cuda:2')

user_prompt = "总结下面的文本，给我列出一份带有主要见解和最重要事实的要点清单。字数小于250字，不要太多。"
system_prompt = "你是一名专业的投研分析人员，你的任务是整理出投研分析所需要的内容,先总结，然后发布评论，如果无法总结，请回答不知道。评论内容在总结内容之后，以【评论】开头。全部内容不要超过150字,不允许分段。"

with open('index_300_keynotes.jsonl','w',encoding='utf-8') as f:
    for i , line in df_news.iterrows():
        if line['content'] == '':
            continue
        news = line['content'][0:4000]
        messages = [
                    

                    {"role": "user", "content": f"{user_prompt}\n{news}"},
                    {"role":"system", "content":f"{system_prompt}"},

                ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt",truncation='only_first')
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=500)
        key_note = llm.generate(prompt,sampling_params)[0].outputs[0].text
        line['key_note'] = key_note
        #cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        # trust_remote_code is needed to use the encode method
        embeddings = model.encode([key_note, news])
        line['embeddings'] = embeddings
        #print(cos_sim(embeddings[0], embeddings[1]))
        f.write(line.to_json(force_ascii=False))
        f.write("\n")
        #json.dump(line.to_json(), f, ensure_ascii=False)

f.close()