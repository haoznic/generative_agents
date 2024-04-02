"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import numpy as np
# import openai
import time
from numpy import dot
from numpy.linalg import norm
from utils import *
from transformers import GPT2TokenizerFast

# openai.api_key = random.choice(openai_api_key)

# from langchain_community.llms import Tongyi
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch 
from torch import nn
from os.path import join as p_join
# set os path 
import os
# import sys
# sys.path.append('../../../')

# with open("../config.json","r") as fp:
#     settings = json.load(fp)
#     os.environ["DASHSCOPE_API_KEY"] = settings["DASHSCOPE_API_KEY"]  

def temp_sleep(seconds=0.1):
    time.sleep(seconds)

    

class MyQwen():
    def __init__(self):
        self.device = "cuda" # the device to load the model onto
        self.modelpath ="/mnt/workspace/project/llm/weights/qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.modelpath,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelpath)

        self.emb_dir ="/mnt/workspace/project/llm/weights/qwen/Qwen-7B-Embedding"
        self.emb_model_path = self.emb_dir+"/qwen_7b_embed.pth"
        self.embed_dict = torch.load(self.emb_model_path)
        vocab_size, embd_dim = self.embed_dict['weight'].size()
        self.embed = nn.Embedding(vocab_size, embd_dim)
        self.embed.load_state_dict(self.embed_dict)

    def invoke(self, prompt: object, gpt_parameter:object=None) -> object:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        if(gpt_parameter!=None):
            # t = np.max([0.1, gpt_parameter["temperature"]])
            t = 0.1 if gpt_parameter["temperature"]==0 else gpt_parameter["temperature"]
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=gpt_parameter["max_tokens"],
                temperature=t,
                top_p=gpt_parameter["top_p"],
            )
        else:
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
    #     messages=[{"role": "user", "content": prompt}],
    # #         temperature=gpt_parameter["temperature"],
    # #         max_tokens=gpt_parameter["max_tokens"],
    # #         top_p=gpt_parameter["top_p"],
    # #         frequency_penalty=gpt_parameter["frequency_penalty"],
    # #         presence_penalty=gpt_parameter["presence_penalty"],
    # #         stream=gpt_parameter["stream"],
    # #         stop=gpt_parameter["stop"], )
       
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 后处理
        if("Activity: " in response):
            response = response.split("Activity: ")[1].split("[(ID:")[0]
        
        # 'stop': ['\n']
        if(gpt_parameter!=None):
            stop=gpt_parameter["stop"]
            if(stop!=None):
                if(isinstance(stop, list)):
                    stop=stop[0]
                response = response.split(stop)[0]
            
        if(response.endswith("\n[(")):
            response = response.split("\n[(")[0]
        
        
        return response
    
    def embedding(self, text: object) -> object:
        text = text.replace("\n", " ")
        if not text:
            text = "this is blank"

        ipts = self.tokenizer(text, return_tensors='pt')['input_ids']
        emb_src = self.embed(ipts).detach().numpy()
        emb = np.mean(emb_src, axis=1).flatten()*0.6+np.max(emb_src, axis=1).flatten()*0.4

        # print("emb------------:",emb)
        return emb
    def cos_sim(self, sentence_embedding1, sentence_embedding2):
        return dot(sentence_embedding1, sentence_embedding2)/(norm(sentence_embedding1)*norm(sentence_embedding2))

llm =  MyQwen()
    
def Tongyi_request(prompt: object) -> object:
    temp_sleep()

    try:
        response = llm.invoke(prompt)

#     prompt = "Give me a short introduction to large language model."
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
    
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


        
        
        # # llm call
        # llm = Tongyi()
        # # llm.model_name = 'qwen-max'
        # llm.model_name ='qwen-max-longcontext'
        # # llm.model_name = 'qwen-plus'
        # messages=json.dumps([{"role": "user", "content": prompt}])
        # print("messages: [",messages,"]")
        # # print("os.environ[DASHSCOPE_API_KEY]:",os.environ["DASHSCOPE_API_KEY"])
        # response = llm.invoke(messages)
        
        
        
        # 后处理
        if(response.strip().startswith("```json")):
            response = response.strip().split("```json")[1].strip().split("```")[0].strip()
        if(response.strip().startswith("['")):
            response = response.strip().split("['")[1].strip().split("']")[0].strip()

        print("response: [",response,"]")
        return response
    except Exception as e:
        return f"TongYi ERROR:{e}"


def ChatGPT_single_request(prompt):
    return Tongyi_request(prompt)
    # temp_sleep()
    # openai.api_key = random.choice(openai_api_key)

    # completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response. 
    ARGS:
        prompt: a str prompt
        gpt_parameter: a python dictionary with the keys indicating the names of  
                    the parameter and the values indicating the parameter 
                    values.   
    RETURNS: 
        a str of GPT-3's response. 
    """
    return Tongyi_request(prompt)


    # temp_sleep()

    # try:
    #     completion = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return completion["choices"][0]["message"]["content"]

    # except:
    #     print("ChatGPT ERROR")
    #     return "ChatGPT ERROR"


def ChatGPT_request(prompt, gpt_parameter={}):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    return Tongyi_request(prompt)
    # # temp_sleep()
    # openai.api_key = random.choice(openai_api_key)

    # try:
    #     completion = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return completion["choices"][0]["message"]["content"]

    # except Exception as e:
    #     print(f"ChatGPT ERROR{e}")
    #     return f"ChatGPT ERROR{e}"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):

        try:
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass

    return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
    # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    # openai.api_key = random.choice(openai_api_key)

    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'
    prompt += "\n只需要输出 output json，不需要别的内容"

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):

        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()

            if(isinstance(json.loads(curr_gpt_response),list)):
                print("curr_gpt_response is list")
                curr_gpt_response = json.loads(curr_gpt_response)
            elif(isinstance(json.loads(curr_gpt_response),dict)):
                print("curr_gpt_response is dict")
                end_index = curr_gpt_response.rfind('}') + 1
                curr_gpt_response = curr_gpt_response[:end_index]
                end_index,curr_gpt_response
                curr_gpt_response = json.loads(curr_gpt_response)["output"]
            else:
                print("curr_gpt_response is ",type(curr_gpt_response))
                end_index = curr_gpt_response.rfind('}') + 1
                curr_gpt_response = curr_gpt_response[:end_index]
                curr_gpt_response = json.loads(curr_gpt_response)["output"]

            # print ("---ashdfaf")
            # print (curr_gpt_response)
            # print ("000asdfhia")

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            time.sleep(1)
            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass

    return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    # openai.api_key = random.choice(openai_api_key)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")
                time.sleep(1)
        except:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

# def ask(question):
#     llm = Tongyi()
#     out  = llm.invoke(question)
#     return out

def GPT_request(prompt, gpt_parameter):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    temp_sleep()

    try:
        print(f"messages 1:[{prompt}]")
        print(f"gpt_parameter 1:[{gpt_parameter}]")
        response = llm.invoke(prompt,gpt_parameter)
        print(f"response 1:[{response}]")
        # 后处理
        if(response.strip().startswith("```json")):
            response = response.strip().split("```json")[1].strip().split("```")[0].strip()

        return response
    except Exception as e:
        print(f"TOKEN LIMIT EXCEEDED: {e}")
        return f"TOKEN LIMIT EXCEEDED"


    # temp_sleep()
    # openai.api_key = random.choice(openai_api_key)

    # try:
    #     response = openai.ChatCompletion.create(
    #         model=gpt_parameter["engine"],
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=gpt_parameter["temperature"],
    #         max_tokens=gpt_parameter["max_tokens"],
    #         top_p=gpt_parameter["top_p"],
    #         frequency_penalty=gpt_parameter["frequency_penalty"],
    #         presence_penalty=gpt_parameter["presence_penalty"],
    #         stream=gpt_parameter["stream"],
    #         stop=gpt_parameter["stop"], )
    #     time.sleep(1)
    #     return response.choices[0]['message']['content']
    # except Exception as e:
    #     print(f"TOKEN LIMIT EXCEEDED: {e}")
    #     return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
    """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    # print(f"curr_input:[{curr_input}]")
    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        # print(f"count:{count}, i:{i}")
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
        time.sleep(1)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    if verbose:
        print(prompt)
    # openai.api_key = random.choice(openai_api_key)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        time.sleep(1)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
    emb = llm.embedding(text)
    return emb

# def get_embedding(text, model="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     if not text:
#         text = "this is blank"

#     tokenizer = GPT2TokenizerFast.from_pretrained('../../models/text-embedding-ada-002')
#     emb = tokenizer.encode(text)
#     print("emb------------:",emb)
#     return emb


# def get_embedding(text, model="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     if not text:
#         text = "this is blank"
#     return openai.Embedding.create(
#         input=[text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
    gpt_parameter = {"engine": "qwen-max", "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)


    def __func_validate(gpt_response):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True


    def __func_clean_up(gpt_response):
        cleaned_response = gpt_response.strip()
        return cleaned_response


    output = safe_generate_response(prompt,
                                    gpt_parameter,
                                    5,
                                    "rest",
                                    __func_validate,
                                    __func_clean_up,
                                    True)

    print(output)
