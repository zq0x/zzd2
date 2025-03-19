from dataclasses import dataclass, fields
import gradio as gr
import redis
import threading
import time
import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import huggingface_hub
from huggingface_hub import snapshot_download
import logging
import psutil

vllm_supported_architectures = [
    "aquilaforcausallm",
    "aquilamodel",    
    "arcticforcausallm",
    "baichuanforcausallm",
    "bloomforcausallm",
    "bartforconditionalgeneration",
    "chatglmmodel",
    "cohereforcausallm",
    "cohere2forcausallm",
    "dbrxforcausallm",
    "decilmforcausallm",
    "deepseekforcausallm",
    "deepseekv2forcausallm",
    "deepseekv3forcausallm",
    "exaoneforcausallm",
    "falconforcausallm",
    "falconmambaforcausallm",
    "gemmaforcausallm",
    "gemma2forcausallm",
    "gemma3forcausallm",
    "glmforcausallm",
    "gpt2lmheadmodel",
    "gptbigcodeforcausallm",
    "gptjforcausallm",
    "gptneoxforcausallm",
    "graniteforcausallm",
    "granitemoeforcausallm",
    "granitemoessharedforcausallm",
    "gritlm",
    "grok1modelforcausallm",
    "internlmforcausallm",
    "internlm2forcausallm",
    "internlm3forcausallm",
    "jaislmheadmodel",
    "jambaforcausallm",
    "llamaforcausallm",
    "mambaforcausallm",
    "mistralmodel",
    "minicpmforcausallm",
    "minicpm3forcausallm",
    "mistralforcausallm",
    "mixtralforcausallm",
    "mptforcausallm",
    "nemotronforcausallm",
    "olmoforcausallm",
    "olmo2forcausallm",
    "olmoeforcausallm",
    "optforcausallm",
    "orionforcausallm",
    "phiforcausallm",
    "phi3forcausallm",
    "phi3smallforcausallm",
    "phimoeforcausallm",
    "persimmonforcausallm",
    "qwenlmheadmodel",
    "qwen2forcausallm",
    "qwen2moeforcausallm",
    "stablelmalphaforcausallm",    
    "stablelmforcausallm",
    "starcoder2forcausallm",
    "solarforcausallm",
    "telechat2forcausallm",
    "xverseforcausallm",
    "bertmodel",
    "gemma2model",
    "llamamodel",
    "qwen2model",
    "robertamodel",
    "xlmrobertamodel",
    "internlm2forrewardmodel",
    "qwen2forrewardmodel",
    "qwen2forprocessrewardmodel",
    "jambaforsequenceclassification",
    "qwen2forsequenceclassification",
    "bertforsequenceclassification",
    "robertaforsequenceclassification",
    "xlmrobertaforsequenceclassification",
    "ariaforconditionalgeneration",
    "blip2forconditionalgeneration",
    "chameleonforconditionalgeneration",
    "deepseekvlv2forcausallm",
    "florence2forconditionalgeneration",
    "fuyuforcausallm",
    "gemma3forconditionalgeneration",
    "glm4vforcausallm",
    "h2ovlchatmodel",
    "idefics3forconditionalgeneration",
    "internvlchatmodel",
    "llavaforconditionalgeneration",
    "llavanextforconditionalgeneration",
    "llavanextvideoforconditionalgeneration",
    "llavaonevisionforconditionalgeneration",
    "minicpmo",
    "minicpmv",
    "mllamaforconditionalgeneration",
    "molmoforcausallm",
    "nvlm_d_model",
    "paligemmaforconditionalgeneration",
    "phi3vforcausallm",
    "phi4mmforcausallm",
    "pixtralforconditionalgeneration",
    "qwenvlforconditionalgeneration",
    "qwen2audioforconditionalgeneration",
    "qwen2vlforconditionalgeneration",
    "qwen2_5_vlforconditionalgeneration",
    "ultravoxmodel",
    "llavanextforconditionalgeneration",
    "phi3vforcausallm",
    "qwen2vlforconditionalgeneration",
    "whisper"
]

REQUEST_TIMEOUT = 50
def wait_for_backend(backend_url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(backend_url, json={"req_method": "list"}, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Backend container is online.")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Backend is not yet reachable
        time.sleep(5)  # Wait for 5 seconds before retrying
    print(f"Timeout: Backend container did not come online within {timeout} seconds.")
    return False

docker_container_list = []
current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

try:
    r = redis.Redis(host="redis", port=6379, db=0)
    db_gpu = json.loads(r.get('db_gpu'))
    # print(f'db_gpu: {db_gpu} {len(db_gpu)}')
    db_gpu_data_len = len(db_gpu_data)
except Exception as e:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

LOG_PATH= './logs'
LOGFILE_CONTAINER = './logs/logfile_container_frontend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f' [START] started logging in {LOGFILE_CONTAINER}')

def load_log_file(req_container_name):
    print(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    logging.info(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    try:
        with open(f'{LOG_PATH}/logfile_{req_container_name}.log', "r", encoding="utf-8") as file:
            lines = file.readlines()
            last_20_lines = lines[-20:]  # Get the last 20 lines
            reversed_lines = last_20_lines[::-1]  # Reverse them
            return ''.join(reversed_lines)
    except Exception as e:
        return f'{e}'












def get_container_data():
    try:
        res_container_data_all = json.loads(r.get('db_container'))
        return res_container_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
   
def get_network_data():
    try:
        res_network_data_all = json.loads(r.get('db_network'))
        return res_network_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(r.get('db_gpu'))
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_docker_container_list():
    global docker_container_list
    response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"list"})
    # print(f'[get_docker_container_list] response: {response}')
    res_json = response.json()
    # print(f'[get_docker_container_list] res_json: {res_json}')
    docker_container_list = res_json.copy()
    if response.status_code == 200:
        # print(f'[get_docker_container_list] res = 200')
        return res_json
    else:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        # logging.info(f'[get_docker_container_list] [get_docker_container_list] res_json: {res_json}')
        return f'Error: {response.status_code}'

def docker_api_logs(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"logs","req_model":req_model})
        res_json = response.json()
        return ''.join(res_json["result_data"])
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop'

def docker_api_network(req_container_name):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"network","req_container_name":req_container_name})
        res_json = response.json()
        if res_json["result"] == 200:
            return f'{res_json["result_data"]["networks"]["eth0"]["rx_bytes"]}'
        else:
            return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {res_json}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {e}'
    
def docker_api_start(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"start","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action start {e}'

def docker_api_stop(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"stop","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop {e}'

def docker_api_delete(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"delete","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action delete {e}'

def docker_api_create(req_model, req_pipeline_tag, req_port_model, req_port_vllm):
    try:
        req_container_name = str(req_model).replace('/', '_')
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"create","req_container_name":req_container_name,"req_model":req_model,"req_runtime":"nvidia","req_port_model":req_port_model,"req_port_vllm":req_port_vllm})
        response_json = response.json()
        
        new_entry = [{
            "gpu": 8,
            "path": f'/home/cloud/.cache/huggingface/{req_model}',
            "container": "0",
            "container_status": "0",
            "running_model": req_container_name,
            "model": req_model,
            "pipeline_tag": req_pipeline_tag,
            "port_model": req_port_model,
            "port_vllm": req_port_vllm
        }]
        r.set("db_gpu", json.dumps(new_entry))

        print(response_json["result"])
        if response_json["result"] == 200:
            return f'{response_json["result_data"]}'
        else:
            return f'Create result ERR no container_id: {str(response_json)}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'error docker_api_create'

def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        if len(model_ids) < 1:
            model_ids = ["No models found!"]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes * 2
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

def get_info(selected_id):    
    global current_models_data    
    res_model_data = {
        "search_data" : "",
        "model_id" : selected_id,
        "pipeline_tag" : "",
        "architectures" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }

    try:
        for item in current_models_data:
            if item['id'] == selected_id:
                
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "architectures" in item:
                    res_model_data["architectures"] = item["architectures"][0]
                                                    
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
                
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"]

def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "hf_data_config" : "",
            "config_data" : "",
            "architectures" : "",
            "model_type" : "",
            "quantization" : "",
            "tokenizer_config" : "",
            "model_id" : selected_id,
            "size" : 0,
            "gated" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "config" in model_info.__dict__:
                    res_model_data['hf_data_config'] = model_info_json["config"]
                    if "architectures" in model_info_json["config"]:
                        res_model_data['architectures'] = model_info_json["config"]["architectures"][0]
                    if "model_type" in model_info_json["config"]:
                        res_model_data['model_type'] = model_info_json["config"]["model_type"]
                    if "tokenizer_config" in model_info_json["config"]:
                        res_model_data['tokenizer_config'] = model_info_json["config"]["tokenizer_config"]
                               
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:
                    print(f'  FOUND safetensors')
                    logging.info(f'  GFOUND safetensors')   
                    
                    safetensors_json = vars(model_info.safetensors)
                    
                    
                    print(f'  FOUND safetensors:::::::: {safetensors_json}')
                    logging.info(f'  GFOUND safetensors:::::::: {safetensors_json}') 
                    try:
                        first_key = next(iter(safetensors_json['parameters'].keys()))
                        print(f'  FOUND first key in parameters:::::::: {first_key}')
                        res_model_data['quantization'] = first_key
                        
                    except Exception as get_model_info_err:
                        print(f'  first key NOT FOUND in parameters:::::::: {first_key}')
                        pass
                    
                    print(f'  FOUND safetensors TOTAL :::::::: {safetensors_json["total"]}')
                    logging.info(f'  GFOUND safetensors:::::::: {safetensors_json["total"]}')   
                    res_model_data['size'] = int(safetensors_json["total"]) * 2
                else:
                    print(f' !!!!DIDNT FIND safetensors !!!! :::::::: ')
                    logging.info(f' !!!!!! DIDNT FIND safetensors !!:::::::: ') 
            
            
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                url = f'https://huggingface.co/{selected_id}/resolve/main/config.json'
                response = requests.get(url)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json
                    if "architectures" in res_model_data["config_data"]:
                        res_model_data["architectures"] = res_model_data["config_data"]["architectures"][0]
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            if res_model_data["size"] == 0:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [get_additional_info] res_model_data["size"] == 0 ...')
                logging.info(f' **************** [get_additional_info] res_model_data["size"] == 0...')
                try:
                    res_model_data["size"] = calculate_model_size(res_model_data["config_data"]) 
                except Exception as get_config_json_err:
                    res_model_data["size"] = 0
    
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["architectures"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"]
        
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"]

def gr_load_check(selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization):
    
    global vllm_supported_architectures
    
    
    # check CUDA support mit backend call
    
    # if "gguf" in selected_model_id.lower():
    #     return f'Selected a GGUF model!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    req_model_storage = "/models"
    req_model_path = f'{req_model_storage}/{selected_model_id}'
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path}) ...')
    logging.info(f' **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path})...')
    


    models_found = []
    try:                   
        if os.path.isdir(req_model_storage):
            print(f' **************** found model storage path! {req_model_storage}')
            print(f' **************** getting folder elements ...')       
            logging.info(f' **************** found model storage path! {req_model_storage}')
            logging.info(f' **************** getting folder elements ...')                        
            for m_entry in os.listdir(req_model_storage):
                m_path = os.path.join(req_model_storage, m_entry)
                if os.path.isdir(m_path):
                    for item_sub in os.listdir(m_path):
                        sub_item_path = os.path.join(m_path, item_sub)
                        models_found.append(sub_item_path)        
            print(f' **************** found models ({len(models_found)}): {models_found}')
            logging.info(f' **************** found models ({len(models_found)}): {models_found}')
        else:
            print(f' **************** found models ({len(models_found)}): {models_found}')
            logging.info(f' **************** ERR model path not found! {req_model_storage}')
    except Exception as e:
        logging.info(f' **************** ERR getting models in {req_model_storage}: {e}')

    
    logging.info(f' **************** does requested model path match downloaded?')
    model_path = selected_model_id
    if req_model_path in models_found:
        print(f' **************** FOUND MODELS ALREADY!!! {selected_model_id} ist in {models_found}')
        model_path = req_model_path
        return f'Model already downloaded!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:
        print(f' **************** NUH UH DIDNT FIND MODEL YET!! {selected_model_id} ist NAWT in {models_found}')
    
    
        
    if selected_model_architectures == '':
        return f'Selected model has no architecture', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] selected_model_architectures.lower() : {selected_model_architectures.lower()}')
    logging.info(f' **************** [gr_load_check] selected_model_architectures.lower() : {selected_model_architectures.lower()}')

    if selected_model_architectures.lower() not in vllm_supported_architectures:
        if selected_model_transformers != 'True':   
            return f'Selected model architecture is not supported by vLLM but transformers are available (you may try to load the model in gradio Interface)', gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return f'Selected model architecture is not supported by vLLM and has no transformers', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)     
    
    if selected_model_pipeline_tag == '':
        return f'Selected model has no pipeline tag', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
    if selected_model_pipeline_tag not in ["text-generation","automatic-speech-recognition"]:
        return f'Only "text-generation" and "automatic-speech-recognition" models supported', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    if selected_model_private != 'False':        
        return f'Selected model is private', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_gated != 'False':        
        return f'Selected model is gated', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_transformers != 'True':        
        return f'Selected model has no transformers', gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        
    if selected_model_size == '0':        
        return f'Selected model has no size', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    return f'Selected model is supported by vLLM!', gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def network_to_pd():       
    rows = []
    try:
        network_list = get_network_data()
        # logging.info(f'[network_to_pd] network_list: {network_list}')  # Use logging.info instead of logging.exception
        for entry in network_list:
            # entry_info = ast.literal_eval(entry['info'])  # Parse the string into a dictionary
            # print("entry_info")
            # print(entry_info)
            # entry_info_networks = entry_info.get("networks", {})
            # print("entry_info_networks")
            # print(entry_info_networks)            
            # first_network_key = next(iter(entry_info_networks))
            # print("first_network_key")
            # print(first_network_key)                 
            # first_network_data = entry_info_networks[first_network_key]
            # print("first_network_data")
            # print(first_network_data)                          
            # first_network_data_rx_bytes = first_network_data["rx_bytes"]
            # print("first_network_data_rx_bytes")
            # print(first_network_data_rx_bytes)              
            # print(f"First network data: {first_network_data}")
            
            # rows.append({
            #     "container": entry["container"],
            #     "current_dl": entry["current_dl"],
            #     "timestamp": entry["timestamp"],
            #     "info": entry["info"]
            # })
               
            rows.append({
                "container": entry["container"],
                "current_dl": entry["current_dl"]
            })
            
            
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "container": "0",
                "current_dl": f'0',
                "timestamp": f'0',
                "info": f'0'
        })
        df = pd.DataFrame(rows)
        return df

def container_to_pd():       
    rows = []
    try:
        container_list = get_container_data()
        # print("container_list")
        # print(container_list)
        # logging.info(f'[container_to_pd] container_list: {container_list}')  # Use logging.info instead of logging.exception
        for entry in container_list:
            container_info = ast.literal_eval(entry['container_info'])  # Parse the string into a dictionary
            rows.append({
                "container_i": entry["container_i"]
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "network_i": "0"
        })
        df = pd.DataFrame(rows)
        return df

def gpu_to_pd():       
    rows = []
    try:
        gpu_list = get_gpu_data()
        # logging.info(f'[gpu_to_pd] gpu_list: {gpu_list}')  # Use logging.info instead of logging.exception
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry['gpu_info'])  # Parse the string into a dictionary
            rows.append({                
                "current_uuid": gpu_info.get("current_uuid", "0"),
                "gpu_i": entry.get("gpu_i", "0"),
                "gpu_util": gpu_info.get("gpu_util", "0"),
                "mem_util": gpu_info.get("mem_util", "0"),
                "temperature": gpu_info.get("temperature", "0"),
                "fan_speed": gpu_info.get("fan_speed", "0"),
                "power_usage": gpu_info.get("power_usage", "0"),
                "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                "clock_info_mem": gpu_info.get("clock_info_mem", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "cuda_cores": gpu_info.get("cuda_cores", "0"),
                "compute_capability": gpu_info.get("compute_capability", "0"),
                "supported": gpu_info.get("supported", "0"),
                "not_supported": gpu_info.get("not_supported", "0"),
                "status": "ok"
            })

        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

gpu_to_pd()


VLLM_URL = f'http://container_vllm:{os.getenv("VLLM_PORT")}/vllmt'


# def vllm_api(request: gr.Request): 
# def vllm_api(req_type,max_tokens=None,temperature=None,prompt_in=None):


def vllm_api(
                req_type,
                model=None,
                pipeline_tag=None,
                max_model_len=None,
                enforce_eager=None,
                enable_prefix_caching=None,
                pipeline_parallel_size=None,
                tensor_parallel_size=None,
                max_parallel_loading_workers=None,
                kv_cache_dtype=None,
                port=None,
                swap_space=None,                
                gpu_memory_utilization=None,
                enable_chunked_prefill=None,
                trust_remote_code=None,
                load_format=None,
                dtype=None,
                quantization_param_path=None,
                block_size=None,
                num_lookahead_slots=None,
                seed=None,
                num_gpu_blocks_override=None,
                max_num_batched_tokens=None,
                max_num_seqs=None,
                max_logprobs=None,
                quantization=None,
                max_content_len_to_capture=None,
                tokenizer_pool_size=None,
                tokenizer_pool_type=None,
                tokenizer_pool_extra_config=None,
                enable_lora=None,
                max_loras=None,
                max_lora_rank=None,
                lora_extra_vocab_size=None,
                lora_dtype=None,
                max_cpu_loras=None,
                device=None,
                image_input_type=None,
                image_token_id=None,
                image_input_shape=None,
                image_feature_size=None,
                scheduler_delay_factor=None,
                speculative_model=None,
                num_speculative_tokens=None,
                speculative_max_model_len=None,
                model_loader_extra_config=None,
                engine_use_ray=None,
                disable_log_requests=None,
                max_log_len=None,
                top_p=None,
                temperature=None,
                max_tokens=None,
                prompt_in=None
             ):
    
    try:

        FALLBACK_VLLM_API = {}
        logging.info(f' [vllm_api] req_type: {req_type}')
        logging.info(f' [vllm_api] model: {model}')
        logging.info(f' [vllm_api] pipeline_tag: {pipeline_tag}')
        logging.info(f' [vllm_api] max_model_len: {max_model_len}')
        logging.info(f' [vllm_api] tensor_parallel_size: {tensor_parallel_size}')
        logging.info(f' [vllm_api] gpu_memory_utilization: {gpu_memory_utilization}')
        logging.info(f' [vllm_api] top_p: {top_p}')
        logging.info(f' [vllm_api] temperature: {temperature}')
        logging.info(f' [vllm_api] max_tokens: {max_tokens}')
        logging.info(f' [vllm_api] prompt_in: {prompt_in}')

        if req_type == "load":
            response = "if you see this it didnt work :/"  
            
            logging.info(f' [vllm_api] [{req_type}] gpu_memory_utilization: {gpu_memory_utilization}')
            response = requests.post(VLLM_URL, json={
                "req_type":"load",
                "max_model_len":int(max_model_len),
                "tensor_parallel_size":int(tensor_parallel_size),
                "gpu_memory_utilization":float(gpu_memory_utilization),
                "model":str(model)
            })
            if response.status_code == 200:
                logging.info(f' [vllm_api] [{req_type}] status_code: {response.status_code}') 
                response_json = response.json()
                logging.info(f' [vllm_api] [{req_type}] response_json: {response_json}') 
                response_json["result_data"] = response_json["result_data"]
                return response_json["result_data"]                
            else:
                logging.info(f' [vllm_api] [{req_type}] response: {response}')
                FALLBACK_VLLM_API["result_data"] = f'{response}'
                return FALLBACK_VLLM_API
    
    

        if req_type == "generate":
            response = "if you see this it didnt work :/"  
            logging.info(f' [vllm_api] [{req_type}] temperature: {temperature}')
            response = requests.post(VLLM_URL, json={
                "req_type":"generate",
                "prompt":str(prompt_in),
                "temperature":float(temperature),
                "top_p":float(top_p),
                "max_tokens":int(max_tokens)
            })
            if response.status_code == 200:
                logging.info(f' [vllm_api] [{req_type}] status_code: {response.status_code}') 
                response_json = response.json()
                logging.info(f' [vllm_api] [{req_type}] response_json: {response_json}') 
                return response_json["result_data"]                
            else:
                logging.info(f' [vllm_api] response: {response}')
                FALLBACK_VLLM_API["result_data"] = f'{response}'
                return FALLBACK_VLLM_API
    
    
    except Exception as e:
        logging.exception(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] Exception occured: {e}', exc_info=True)
        FALLBACK_VLLM_API["result_data"] = f'{e}'
        return FALLBACK_VLLM_API






def refresh_container():
    try:
        global docker_container_list
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method": "list"})
        docker_container_list = response.json()
        return docker_container_list
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'err {str(e)}'



@dataclass
class VllmInputComponents:
    model_id: gr.Textbox
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmInputValues:    
    model_id: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int

# Define the InputComponents and InputValues classes
@dataclass
class InputComponents:
    param1: gr.Slider
    param2: gr.Number
    quantity: gr.Slider
    animal: gr.Dropdown
    countries: gr.CheckboxGroup
    place: gr.Radio
    activity_list: gr.Dropdown
    morning: gr.Checkbox
    param0: gr.Textbox
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class InputValues:    
    param1: int
    param2: int
    quantity: int
    animal: str
    countries: list
    place: str
    activity_list: list
    morning: bool
    param0: str







BACKEND_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest'

def docker_api(req_type,req_model=None,req_task=None,req_prompt=None,req_temperature=None, req_config=None):
    
    try:
        print(f'got model_config: {req_config} ')
        response = requests.post(BACKEND_URL, json={
            "req_type":req_type,
            "req_model":req_model,
            "req_task":req_task,
            "req_prompt":req_prompt,
            "req_temperature":req_temperature,
            "req_model_config":req_config
        })
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json["result_status"] != 200:
                logging.exception(f'[docker_api] Response Error: {response_json["result_data"]}')
            return response_json["result_data"]                
        else:
            logging.exception(f'[docker_api] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e


def toggle_vllm_create_engine_arguments(vllm_list):
    
    if "Create New" in vllm_list:
        return (
            gr.Textbox(visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Textbox(visible=False),
        gr.Button(visible=False)
    )

def load_vllm_running(req_model_id,*params):
    
    
    
    
    try:
        print(f' >>> load_vllm_running req_model_id: {req_model_id} ')
        print(f' >>> load_vllm_running got params: {params} ')
        logging.exception(f'[load_vllm_running] >> req_model_id: {req_model_id} ')
        logging.exception(f'[load_vllm_running] >> got params: {params} ')
                
        req_params = VllmInputValues(*params)


        response = requests.post(BACKEND_URL, json={
            "req_method":"test",
            "model_id":req_params.model_id,
            "max_model_len":req_params.max_model_len,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization
        }, timeout=REQUEST_TIMEOUT)
        res_json = response.json()
        if response.status_code == 200:
            response_json = response.json()
            if response_json["result_status"] != 200:
                logging.exception(f'[load_vllm_running] Response Error: {res_json["result_data"]}')
                return f'ERROR {res_json["result_data"]}'
                
            return f'{res_json["result_data"]}'
        else:
            logging.exception(f'[load_vllm_running] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e
    
    
    



    # result = f"Running Settings: \n max_model_len: {req_params.max_model_len}, \n tensor_parallel_size: {req_params.tensor_parallel_size}, \n gpu_memory_utilization: {req_params.gpu_memory_utilization}"
    
    # response = requests.post(BACKEND_URL, json={"req_method":"test","req_params":json.dumps(req_params)})

    # print(f'[load_vllm_running] response: {response}')
    # logging.info(f' [load_vllm_running] response: {response}')
    # res_json = response.json()
    # print(f'[load_vllm_running] res_json: {res_json}')
    # logging.info(f' [load_vllm_running] res_json: {res_json}')
    # if response.status_code == 200:
    #     print(f'[load_vllm_running] response.status_code == 200')
    #     logging.info(f' [load_vllm_running] response.status_code == 200')
    #     return res_json
    # else:
    #     print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
    #     logging.info(f' [load_vllm_running] {e}')
    #     # logging.info(f'[get_docker_container_list] [get_docker_container_list] res_json: {res_json}')
    #     return f'Error: {response.status_code}'
    

def predict_with_my_model(*params):
    req_params = InputValues(*params)
    result = f"Processed values: {req_params.param1}, {req_params.param2}, {req_params.quantity}, {req_params.animal}, {req_params.countries}, {req_params.place}, {req_params.activity_list}, {req_params.morning}, {req_params.param0}, {req_params.param0}, {req_params.param0}, {req_params.param0}"
    return result





prev_bytes_recv = 0
def get_download_speed():
    try:
        global prev_bytes_recv        
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_kb = download_speed / 1024
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)      
        bytes_received_mb = bytes_recv
        return f'{download_speed} bytes ({download_speed_mbit_s:.2f} MBit/s)'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'




def download_from_hf_hub(selected_model_id):
    try:
        selected_model_id_arr = str(selected_model_id).split('/')
        print(f'selected_model_id_arr {selected_model_id_arr}...')       
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'
        )
        return f'Saved to {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'


download_info_prev_bytes_recv = 0   
download_info_current_model_bytes_recv = 0    
 
def download_info(req_model_size, progress=gr.Progress()):
    global download_info_prev_bytes_recv
    global download_info_current_model_bytes_recv
    download_info_prev_bytes_recv = 0
    download_info_current_model_bytes_recv = 0
    progress(0, desc="Initializing ...")
    progress(0.01, desc="Calculating Download Time ...")
    
    avg_dl_speed_val = 0
    avg_dl_speed = []
    for i in range(0,5):
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2) 
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv
        avg_dl_speed.append(download_speed)
        avg_dl_speed_val = sum(avg_dl_speed)/len(avg_dl_speed)
        logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
        print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  
        time.sleep(1)
    
    logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  



    calc_mean = lambda data: np.mean([x for x in data if (np.percentile(data, 25) - 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))) <= x <= (np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)))]) if data else 0


    avg_dl_speed_val = calc_mean(avg_dl_speed)
        
    
    logging.info(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')    

    est_download_time_sec = int(req_model_size)/int(avg_dl_speed_val)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    est_download_time_sec = int(est_download_time_sec)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')


    for i in range(0,est_download_time_sec):
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv

        progress_percent = (i + 1) / est_download_time_sec
        progress(progress_percent, desc=f"Downloading ... {download_speed_mbit_s:.2f} MBit/s {download_info_current_model_bytes_recv}/{req_model_size}")

        time.sleep(1)
    logging.info(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec/10}')
    print(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec/10}')
    yield f'Progress: 100%\nFiniiiiiiiish!'


def parallel_download(selected_model_size, model_dropdown):
    # Create threads for both functions
    thread_info = threading.Thread(target=download_info, args=(selected_model_size,))
    thread_hub = threading.Thread(target=download_from_hf_hub, args=(model_dropdown,))

    # Start both threads
    thread_info.start()
    thread_hub.start()

    # Wait for both threads to finish
    thread_info.join()
    thread_hub.join()

    return "Download finished!"



def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(10)
    progress(0.05, desc="Initializing...")

    new_string = ""
    total_letters = len(word)
    for i, letter in enumerate(word):
        time.sleep(5.25)
        new_string = letter + new_string
        progress_percent = (i + 1) / total_letters
        progress(progress_percent, desc=f"Reversing... ({int(progress_percent * 100)}%)")
        yield f"Progress: {int(progress_percent * 100)}%\nReversed so far: {new_string}"

    yield f"Progress: 100%\nReversed word: {new_string}"

def create_app():
    with gr.Blocks() as app:
        gr.Markdown(
            """
            # Welcome!
            Select a Hugging Face model and deploy it on a port
            
            **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_
            or selected a tested model from here
            """)

        input_search = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
        btn_search = gr.Button("Search")

        with gr.Row(visible=False) as row_model_select:
            model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False)
        with gr.Row(visible=False) as row_model_info:
            with gr.Column(scale=4):
                with gr.Accordion(("Model Parameters"), open=False):                    
                    with gr.Row():
                        selected_model_id = gr.Textbox(label="id")
                        selected_model_container_name = gr.Textbox(label="container_name")
                        
                        
                    with gr.Row():
                        selected_model_architectures = gr.Textbox(label="architectures")
                        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag")
                        selected_model_transformers = gr.Textbox(label="transformers")
                        
                        
                    with gr.Row():
                        selected_model_model_type = gr.Textbox(label="model_type")
                        selected_model_quantization = gr.Textbox(label="quantization")
                        selected_model_size = gr.Textbox(label="size")
                                                       
                        
                    with gr.Row():
                        selected_model_private = gr.Textbox(label="private")
                        selected_model_gated = gr.Textbox(label="gated")
                        selected_model_downloads = gr.Textbox(label="downloads")
                                          
                        
                        
                    
                    with gr.Accordion(("Model Configs"), open=False):
                        with gr.Row():
                            selected_model_search_data = gr.Textbox(label="search_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_hf_data = gr.Textbox(label="hf_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_config_data = gr.Textbox(label="config_data", lines=20, elem_classes="table-cell")

                    with gr.Row():
                        port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
                        port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
                        
                
                           
        with gr.Row(visible=False) as column_model_actions:
            with gr.Column(scale=3):
                output = gr.Textbox(label="Output", show_label=True, visible=True) 
                download_info_output = gr.Textbox(label="Download Info Output", show_label=True, visible=True) 
            with gr.Column(scale=1):
                
                
                input_search.submit(
                    search_models, 
                    input_search, 
                    [model_dropdown]
                ).then(
                    lambda: gr.update(visible=True), 
                    None, 
                    model_dropdown
                )
                
                btn_search.click(
                    search_models, 
                    input_search, 
                    [model_dropdown]
                ).then(
                    lambda: gr.update(visible=True), 
                    None, 
                    model_dropdown
                )

                btn_dl = gr.Button("DOWNLOAD", visible=False)
                btn_parallel = gr.Button("BTN_PARALLEL", visible=True)
                btn_parallel_two = gr.Button("BTN_PARALLEL_TWO", visible=True)

                    
                btn = gr.Button("input_components -> predict_with_my_model")
                btn_vllm_running = gr.Button("vllm_input_components ->  BACKEND")
                # vllm_engine_arguments_show = gr.Button("GENERATE NEW VLLM", variant="primary")
                # vllm_engine_arguments_close = gr.Button("CANCEL")

                
        
        
                    
                
        with gr.Row(visible=False) as vllm_create_engine_arguments_row:
            with gr.Column(scale=4):
                with gr.Accordion(("Create Parameters"), open=True):
                    input_components = InputComponents(
                        param0=gr.Textbox(placeholder="pasdsssda", value="genau", label="Textbox", info="yes a textbox"),
                        param1=gr.Slider(2, 20, value=1, label="Count", info="Choose between 2 and 20"),
                        param2=gr.Number(label="Number Input", value="2", info="Enter a number"),
                        quantity=gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
                        animal=gr.Dropdown(["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"),
                        countries=gr.CheckboxGroup(["USA", "Japan", "Pakistan"], label="Countries", info="Where are they from?"),
                        place=gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?"),
                        activity_list=gr.Dropdown(["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity", info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."),
                        morning=gr.Checkbox(label="Morning", value=True, info="Did they do it in the morning?")
                    )
            with gr.Column(scale=1, visible=False) as vllm_engine_arguments_btn:
                vllm_engine_arguments_show = gr.Button("GENERATE NEW VLLM", variant="primary")
                vllm_engine_arguments_close = gr.Button("CANCEL")
                

    


        
                
                
                
                
                
        with gr.Column(scale=4) as column_select_vllm_engine_arguments:
            with gr.Row(visible=True):
                vllms=gr.Radio(["vLLM1", "vLLM2", "Create New"], value="vLLM1", label="vLLMs", info="Where to deploy?")


                    
            

            
            
            
            
            
            with gr.Row(visible=True) as vllm_running_engine_arguments_row:
                with gr.Column(scale=4):
                    with gr.Accordion(("vLLM Parameters"), open=True):
                        vllm_input_components = VllmInputComponents(
                            model_id=gr.Textbox(placeholder=f'{selected_model_id}', value=f'{selected_model_id}', label="model_id", info="Hugging Face Model ID"),
                            max_model_len=gr.Slider(1024, 8192, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                            tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                            gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                        )
                with gr.Column(scale=1, visible=True) as vllm_running_engine_argumnts_btn:
                    vllm_running_engine_arguments_show = gr.Button("LOAD VLLM", variant="primary")
                    vllm_running_engine_arguments_close = gr.Button("CANCEL")

        
        model_dropdown.change(
            get_info, 
            model_dropdown, 
            [selected_model_search_data,selected_model_id,selected_model_architectures,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]
        ).then(
            get_additional_info, 
            model_dropdown, 
            [selected_model_hf_data, selected_model_config_data, selected_model_architectures,selected_model_id, selected_model_size, selected_model_gated, selected_model_model_type, selected_model_quantization]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_select
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_info
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            column_model_actions
        ).then(
            gr_load_check, [selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization]
            ,
            [output,btn_dl,vllm_running_engine_arguments_row,vllm_create_engine_arguments_row]
        )
            

        vllm_running_engine_arguments_show.click(
            lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, vllm_running_engine_arguments_row]
        )
        
        vllm_running_engine_arguments_close.click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, vllm_running_engine_arguments_row]
        )




            

        vllm_engine_arguments_show.click(
            lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], 
            None, 
            [vllm_engine_arguments_show, vllm_engine_arguments_close, vllm_create_engine_arguments_row]
        )
        
        vllm_engine_arguments_close.click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], 
            None, 
            [vllm_engine_arguments_show, vllm_engine_arguments_close, vllm_create_engine_arguments_row]
        )



        
        btn.click(
            predict_with_my_model,
            input_components.to_list(),
            [output]
        )

        
        btn_vllm_running.click(
            load_vllm_running,
            [model_dropdown,vllm_input_components.to_list()],
            [output]
        )


        vllms.change(
            toggle_vllm_create_engine_arguments,
            vllms,
            [vllm_create_engine_arguments_row, vllm_engine_arguments_btn]
        )


         
        btn_interface = gr.Button("Load Interface",visible=False)
        @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
        def show_split(text_pipeline, text_model):
            if len(text_model) == 0:
                gr.Markdown("Error pipeline_tag or model_id")
            else:
                selected_model_id_arr = str(text_model).split('/')
                print(f'selected_model_id_arr {selected_model_id_arr}...')            
                gr.Interface.from_pipeline(pipeline(text_pipeline, model=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'))

        timer_dl = gr.Timer(1,active=False)
        timer_dl.tick(get_download_speed, outputs=output)    
        
        
        timer_c = gr.Timer(1,active=False)
        timer_c.tick(refresh_container)
                




        
        
        
        
        
        




        
        
        
        
        
        
        
        
        
        
        

        
        
        
       
        load_btn = gr.Button("Load into vLLM (port: 1370)", visible=True)

        
        
        with gr.Row():
            top_p = gr.Textbox(label="top_p", placeholder="0.95", value=0.95, visible=True)
            temperature = gr.Slider(0.1, 1.0, step=0.1, label="temperature", value=0.8, visible=True)
            max_tokens = gr.Number(label="max_tokens", value=150, visible=True)
        
        
        prompt_in = gr.Textbox(placeholder="Ask a question", value="Follow the", label="Query", show_label=True, visible=True)  
        prompt_out = gr.Textbox(placeholder="Result will appear here", label="Output", show_label=True, visible=True)
        prompt_btn = gr.Button("Submit", visible=True)
        
        
        
                
        with gr.Accordion("Slowly Reverse a Word", open=False):
            reverse_input = gr.Textbox(label="Enter a word to reverse", placeholder="Type here...")
            reverse_output = gr.Textbox(label="Progress and Reversed Word")
            reverse_btn = gr.Button("Reverse Slowly")
            reverse_btn.click(
                fn=slowly_reverse,
                inputs=reverse_input,
                outputs=reverse_output
            )
        # load_btn.click(lambda model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in: vllm_api("load", model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in), inputs=[model_dropdown, selected_model_pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in], outputs=prompt_out)
            
        # prompt_btn.click(lambda model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in: vllm_api("generate", model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in), inputs=[model_dropdown, selected_model_pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in], outputs=prompt_out)

        
        gpu_dataframe = gr.Dataframe(label="GPU information")
        gpu_timer = gr.Timer(1,active=True)
        gpu_timer.tick(gpu_to_pd, outputs=gpu_dataframe)
        
        network_dataframe = gr.Dataframe(label="Network information")
        network_timer = gr.Timer(1,active=True)
        network_timer.tick(network_to_pd, outputs=network_dataframe)
        
        container_state = gr.State([])   
        docker_container_list = get_docker_container_list()     
        @gr.render(inputs=container_state)
        def render_container(render_container_list):
            docker_container_list = get_docker_container_list()
            docker_container_list_sys_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_sys_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_vllm_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] not in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_vllm_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] not in ["/container_redis","/container_backend", "/container_frontend"]]

            def refresh_container():
                try:
                    global docker_container_list
                    response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                    docker_container_list = response.json()
                    return docker_container_list
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return f'err {str(e)}'


            with gr.Accordion(f'vLLM | Running {len(docker_container_list_vllm_running)} | Not Running {len(docker_container_list_vllm_not_running)}', open=False):
                gr.Markdown(f'### Running ({len(docker_container_list_vllm_running)})')

                for current_container in docker_container_list_vllm_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                        
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                    with gr.Row():            
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            inputs=[container_name],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            inputs=[container_id],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        stop_btn = gr.Button("Stop", scale=0)
                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                        stop_btn.click(
                            docker_api_stop,
                            inputs=[container_id],
                            outputs=[container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )

                        delete_btn.click(
                            docker_api_delete,
                            inputs=[container_id],
                            outputs=[container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )
                        
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )


                gr.Markdown(f'### Not running ({len(docker_container_list_vllm_not_running)})')

                for current_container in docker_container_list_vllm_not_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                        
                    with gr.Row():
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            inputs=[container_name],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            inputs=[container_id],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        start_btn = gr.Button("Start", scale=0)
                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                        start_btn.click(
                            docker_api_start,
                            inputs=[container_id],
                            outputs=[container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )

                        delete_btn.click(
                            docker_api_delete,
                            inputs=[container_id],
                            outputs=[container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )
                    
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )
                    
            

            with gr.Accordion(f'System | Running {len(docker_container_list_sys_running)} | Not Running {len(docker_container_list_sys_not_running)}', open=False):
                gr.Markdown(f'### Running ({len(docker_container_list_sys_running)})')

                for current_container in docker_container_list_sys_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                        
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                    with gr.Row():            
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            inputs=[container_name],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            inputs=[container_id],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )

                gr.Markdown(f'### Not Running ({len(docker_container_list_sys_not_running)})')

                for current_container in docker_container_list_sys_not_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                        
                    with gr.Row():
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            inputs=[container_name],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            inputs=[container_id],
                            outputs=[container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                    
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )











            
        
        
        
        def refresh_container_list():
            try:
                global docker_container_list
                response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                docker_container_list = response.json()
                return docker_container_list
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'

        
        btn_dl.click(
            download_info, 
            selected_model_size, 
            download_info_output,
            concurrency_limit=5
        ).then(
            lambda: gr.Timer(active=True), 
            None, 
            timer_dl
        ).then(
            lambda: gr.Timer(active=False), 
            None, 
            timer_dl
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            btn_interface
        )

        
        btn_parallel_two.click(
            download_info, 
            selected_model_size, 
            download_info_output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=True), 
            None, 
            timer_dl
        ).then(
            lambda: gr.Timer(active=False), 
            None, 
            timer_dl
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            btn_interface
        )
        
        btn_parallel_two.click(
            download_from_hf_hub, 
            model_dropdown, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=True), 
            None, 
            timer_dl
        ).then(
            lambda: gr.Timer(active=False), 
            None, 
            timer_dl
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            btn_interface
        )

        # btn_parallel.click(
        #     lambda: gr.Timer(active=True), 
        #     None, 
        #     timer_dl
        # ).then(
        #     lambda size, name: (download_info(size), download_from_hf_hub(name)),
        #     [selected_model_size, model_dropdown],
        #     [download_info_output, output],
        #     concurrency_limit=200
        # ).then(
        #     lambda: gr.Timer(active=False), 
        #     None, 
        #     timer_dl
        # ).then(
        #     lambda: gr.update(visible=True), 
        #     None, 
        #     btn_interface
        # )
        
        btn_parallel.click(
            lambda: gr.update(label="Starting download ...", visible=True), 
            None, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=True), 
            None, 
            timer_dl,
            concurrency_limit=15
        ).then(
            parallel_download, 
            [selected_model_size, model_dropdown], 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=False), 
            None, 
            timer_dl,
            concurrency_limit=15
        ).then(
            lambda: gr.update(label="Download finished!"), 
            None, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            btn_interface,
            concurrency_limit=15
        )

        btn_parallel.click(
            lambda: gr.update(
                label="Starting download ...",
                visible=True), 
            None, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=True), 
            None, 
            timer_dl,
            concurrency_limit=15
        ).then(
            download_info, 
            selected_model_size, 
            download_info_output,
            concurrency_limit=15
        ).then(
            download_from_hf_hub, 
            model_dropdown, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.Timer(active=False), 
            None, 
            timer_dl,
            concurrency_limit=15
        ).then(
            lambda: gr.update(label="Download finished!"), 
            None, 
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            btn_interface,
            concurrency_limit=15
        )

        btn_deploy = gr.Button("Deploy", visible=True)
        btn_deploy.click(lambda: gr.update(label="Building vLLM container",visible=True), None, output).then(docker_api_create,inputs=[model_dropdown,selected_model_pipeline_tag,port_model,port_vllm],outputs=output).then(refresh_container, outputs=[container_state]).then(lambda: gr.Timer(active=True), None, timer_dl).then(lambda: gr.update(visible=True), None, btn_interface)





    return app

# Launch the app
if __name__ == "__main__":
    backend_url = f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest'
    
    # Wait for the backend container to be online
    if wait_for_backend(backend_url):
        app = create_app()
        app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))
    else:
        print("Failed to start application due to backend container not being online.")
