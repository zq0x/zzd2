from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import requests
import redis.asyncio as redis
import sys
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import psutil
import logging


DEFAULT_CONTAINER_STATS = {
    'name': '/error_container',
    'id': '0000000000000000000000000000000000000000000000000000000000000000',
    'read': '2025-01-01T00:00:00.000000000Z',
    'preread': '2025-01-01T00:00:00.000000000Z',
    'pids_stats': {
        'current': 0,
        'limit': 0
    },
    'blkio_stats': {
        'io_service_bytes_recursive': None,
        'io_serviced_recursive': None,
        'io_queue_recursive': None,
        'io_service_time_recursive': None,
        'io_wait_time_recursive': None,
        'io_merged_recursive': None,
        'io_time_recursive': None,
        'sectors_recursive': None
    },
    'num_procs': 0,
    'storage_stats': {},
    'cpu_stats': {
        'cpu_usage': {
            'total_usage': 0,
            'usage_in_kernelmode': 0,
            'usage_in_usermode': 0
        },
        'system_cpu_usage': 0,
        'online_cpus': 0,
        'throttling_data': {
            'periods': 0,
            'throttled_periods': 0,
            'throttled_time': 0
        }
    },
    'precpu_stats': {
        'cpu_usage': {
            'total_usage': 0,
            'usage_in_kernelmode': 0,
            'usage_in_usermode': 0
        },
        'system_cpu_usage': 0,
        'online_cpus': 0,
        'throttling_data': {
            'periods': 0,
            'throttled_periods': 0,
            'throttled_time': 0
        }
    },
    'memory_stats': {
        'usage': 0,
        'stats': {
            'active_anon': 0,
            'active_file': 0,
            'anon': 0,
            'anon_thp': 0,
            'file': 0,
            'file_dirty': 0,
            'file_mapped': 0,
            'file_writeback': 0,
            'inactive_anon': 0,
            'inactive_file': 0,
            'kernel_stack': 0,
            'pgactivate': 0,
            'pgdeactivate': 0,
            'pgfault': 0,
            'pglazyfree': 0,
            'pglazyfreed': 0,
            'pgmajfault': 0,
            'pgrefill': 0,
            'pgscan': 0,
            'pgsteal': 0,
            'shmem': 0,
            'slab': 0,
            'slab_reclaimable': 0,
            'slab_unreclaimable': 0,
            'sock': 0,
            'thp_collapse_alloc': 0,
            'thp_fault_alloc': 0,
            'unevictable': 0,
            'workingset_activate': 0,
            'workingset_nodereclaim': 0,
            'workingset_refault': 0
        },
        'limit': 0
    },
    'networks': {
        'eth0': {
            'rx_bytes': 0,
            'rx_packets': 0,
            'rx_errors': 0,
            'rx_dropped': 0,
            'tx_bytes': 0,
            'tx_packets': 0,
            'tx_errors': 0,
            'tx_dropped': 0
        }
    }
}


# print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)

LOG_PATH= './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_backend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')
# print(f'** connecting to pynvml ... ')
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
# print(f'** pynvml found GPU: {device_count}')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] pynvml found GPU: {device_count}')

device_uuids = []
for i in range(0,device_count):
    # print(f'1 i {i}')
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    # print(f'1 handle {handle}')
    current_uuid = pynvml.nvmlDeviceGetUUID(handle)
    device_uuids.append(current_uuid)

# print(f'** pynvml found uuids ({len(device_uuids)}): {device_uuids} ')


DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_frontend = json.load(f)["frontend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')



# def get_vllm_info():
#     container_info = []
#     try:        
#         res_container_list = client.containers.list(all=True)
#         print(f'** [get_container_info] {res_container_list} ({len(res_container_list)})')
#         for res_container_i in range(0,len(res_container_list)):
#             print("res_container_i")
#             print(res_container_i)
#             print("res_container_list[res_container_i]")
#             print(res_container_list[res_container_i])
#             container_info.append({
#                     "container_i": f'{res_container_i}',
#                     "container_info": f'{res_container_i}'
#             })
#         return container_info
#     except Exception as e:
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
#         return container_info


# async def redis_timer_vllm():
#     while True:
#         try:
#             current_container_info = get_container_info()
#             res_db_container = await r.get('db_container')
#             if res_db_container is not None:
#                 db_container = json.loads(res_db_container)
#                 updated_container_data = []
#                 print(f' [container] 1 len(current_gpu_info): {len(current_container_info)}')
#                 for container_i in range(0,len(current_container_info)):
#                     print(f' [container] gpu_i: {container_i}')
#                     update_data = {
#                         "container_i": container_i,
#                         "container_info": str(current_container_info[container_i]),
#                         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#                     }
#                     updated_container_data.append(update_data)
#                     print(f'[container] 1 updated_container_data: {updated_container_data}')
#                 await r.set('db_container', json.dumps(updated_container_data))
#             else:
#                 updated_container_data = []
#                 for container_i in range(0,len(current_container_info)):
#                     update_data = {
#                         "container_i": container_i,
#                         "container_info": str(current_container_info[container_i]),
#                         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#                     }
#                     updated_container_data.append(update_data)
#                     print(f'[container] 2 updated_container_data: {updated_container_data}')
#                 await r.set('db_container', json.dumps(updated_container_data))
#             await asyncio.sleep(1.0)
#         except Exception as e:
#             print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
#             await asyncio.sleep(1.0)
















prev_bytes_recv = 0
def get_download_speed():
    try:
        global prev_bytes_recv
        
        # print(f'trying to get download speed ...')
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_kb = download_speed / 1024
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)      
        bytes_received_mb = bytes_recv

        return f'download_speed_mbit_s {download_speed_mbit_s} bytes_recv {bytes_recv} download_speed {download_speed} download_speed_kb {download_speed_kb} '
        # return f'{download_speed_kb:.2f} KB/s (total: {bytes_received_mb:.2f})'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'





prev_bytes_recv = 0

def get_network_info():
    network_info = []
    try: 
            
        current_total_dl = get_download_speed()
        network_info.append({
            "container": f'all',
            "info": "infoblabalba",            
            "current_dl": f'{current_total_dl}',
            "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        })
        
        
        # print(f'finding all containers ..')
        res_container_list = client.containers.list(all=True)
        
        # print(f'Found {len(res_container_list)} containers!')

        # Iterate through each container
        for container in res_container_list:
            # Get container stats
            container_stats = container.stats(stream=False)
            
            # Extract network information (rx_bytes)
            networks = container_stats.get('networks', {})
            rx_bytes = 0  # Default value if no network data is found
            if networks:
                # Sum rx_bytes across all network interfaces (e.g., eth0, eth1, etc.)
                rx_bytes = sum(network.get('rx_bytes', 0) for network in networks.values())
            
            # Append network information to the list
            network_info.append({
                "container": container.name,
                "info": "infoblabalba",  # Placeholder for additional info
                "current_dl": str(rx_bytes),  # Use the actual rx_bytes value
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Print container stats for debugging
            # print(f'Container: {container.name}')
            # print(f'Stats: {container_stats}')
            # print(f'Network Info: {networks}')
            # print(f'RX Bytes: {rx_bytes}')

        # Now `network_info` contains the network information for all containers
        # print('Network Info:')
        # print(network_info)


        
        # print(f'Found {len(res_container_list)} containers!')
        # rx_bytes = sum(network.get('rx_bytes', 0) for network in networks.values())


        # for container in res_container_list:
        #     container_stats = container.stats(stream=False)
        #     print(f'container_stats {container_stats}')
        #     print(f'container_stats["info"] {container_stats["info"]}')
            
        #     rx_bytes = sum(network.get('rx_bytes', 0) for network in networks.values())
        #     network_info.append({
        #         "container": container.name,
        #         "info": "infoblabalba", 
        #         "current_dl": "1337",
        #         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #     })
            


        
        
        
        # res_container_list = client.containers.list(all=True)
        # print(f'found {len(res_container_list)} containers!')
        # res_container_list_attrs = [container.attrs for container in res_container_list]
        # for c in res_container_list_attrs:
        #     print('c["name"]')
        #     print()
        #     # get container again lol
        #     req_container = client.containers.get(c["name"])
        #     stats = c.stats(stream=False)
        #     print('stats')
        #     print(stats)
        #     network_info.append({
        #         "container": f'{c["name"]}',
        #         "info": f'network_info_blank', 
        #         "current_dl": f'000000000000000',
        #         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #     })
            
        # print(f'got all containers! printing final before responding ({len(network_info)}) ')        
        return network_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_network_info] {e}')
        return network_info


    # gpu_info.append({
    #         "gpu_i": i,
    #         "current_uuid": current_uuid,
    #         "gpu_util": float(gpu_util),
    #         "mem_used": float(mem_used),
    #         "mem_total": float(mem_total),
    #         "mem_util": float(mem_util)
    # })
    
async def redis_timer_network():
    while True:
        try:
            current_network_info = get_network_info()
            res_db_network = await r.get('db_network')
            if res_db_network is not None:
                db_network = json.loads(res_db_network)
                updated_network_data = []
                # print(f' [network] 1 len(current_gpu_info): {len(current_network_info)}')
                for net_info_obj in current_network_info:
                    # print(f' [network] net_info_obj: {net_info_obj}')
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    # print(f'[network] 1 updated_network_data: {updated_network_data}')
                await r.set('db_network', json.dumps(updated_network_data))
            else:
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    # print(f'[network] 2 updated_network_data: {updated_network_data}')
                await r.set('db_network', json.dumps(updated_network_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [redis_timer_network] {e}')
            await asyncio.sleep(1.0)



pynvml.nvmlInit()
def get_gpu_info():
    try:

        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        for i in range(0,device_count):
            current_gpu_info = {}
            current_gpu_info['gpu_int'] = str(i)           
            

            
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            

            
            try:
                res_current_uuid = pynvml.nvmlDeviceGetUUID(handle)
                current_gpu_info['res_current_uuid'] = f'{res_current_uuid}'
            except Exception as e:
                print(f'0 gpu_info {e}')
            
            
            
        
            
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # mem_util = f'{(mem_used / mem_total) * 100} %'
                res_gpu_util = f'{utilization.gpu}%'
                current_gpu_info['res_gpu_util'] = f'{res_gpu_util}'
                
                
                # res_mem_util = f'{utilization.memory}%'
                # current_gpu_info['res_mem_util'] = f'{res_mem_util}'
            except Exception as e:
                print(f'1 gpu_info {e}')

            try: 
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                res_mem_total = f'{mem_info.total / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_total'] = f'{res_mem_total}'
                res_mem_used = f'{mem_info.used / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_used'] = f'{res_mem_used}'
                res_mem_free = f'{mem_info.free / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_free'] = f'{res_mem_free}'
                
                res_mem_util = (float(mem_info.used / 1024**2)/float(mem_info.total / 1024**2)) * 100
                current_gpu_info['res_mem_util'] = f'{"{:.2f}".format(res_mem_util)}% ({res_mem_used}/{res_mem_total})'

            except Exception as e:
                print(f'2 gpu_info {e}')
            
            try:
                # Get GPU temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                res_temperature = f'{temperature}°C'
                current_gpu_info['res_temperature'] = f'{res_temperature}'
            except Exception as e:
                print(f'3 gpu_info {e}')
                
            try:
                # Get GPU fan speed
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                res_fan_speed = f'{fan_speed}%'
                current_gpu_info['res_fan_speed'] = f'{res_fan_speed}'
            except Exception as e:
                print(f'4 gpu_info {e}')


            # try:
            #     # Get GPU power usage
            #     power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
            #     res_power_usage = f'{power_usage / 1000:.2f} W'
            #     current_gpu_info['res_power_usage'] = f'{res_power_usage}'
            # except Exception as e:
            #     print(f'5 gpu_info {e}')
        
        
            try:
                # Get GPU clock speeds
                clock_info_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                res_clock_info_graphics = f'{clock_info_graphics} MHz'
                current_gpu_info['res_clock_info_graphics'] = f'{res_clock_info_graphics}'
            except Exception as e:
                print(f'6 gpu_info {e}')
            
            
            try:
                clock_info_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                res_clock_info_mem = f'{clock_info_mem} MHz'
                current_gpu_info['res_clock_info_mem'] = f'{res_clock_info_mem}'
            except Exception as e:
                print(f'7 gpu_info {e}')
                
            try:
                # Get GPU compute capability (compute_capability)
                cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
                res_cuda_cores = f'{cuda_cores}'
                current_gpu_info['res_cuda_cores'] = f'{res_cuda_cores}'
            except Exception as e:
                print(f'8 gpu_info {e}')

            res_supported = []
            res_not_supported = []
            try:
                # Get GPU compute capability (CUDA cores)
                compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability_str = f'{compute_capability[0]}.{compute_capability[1]}'
                res_compute_capability = f'{compute_capability_str}'

                if float(res_compute_capability) >= 8:
                    res_supported.append('Bfloat16')
                else:
                    res_not_supported.append('Bfloat16')
            except Exception as e:
                print(f'9 gpu_info {e}')
            
            
            res_supported_str = ",".join(res_supported)
            current_gpu_info['res_supported_str'] = f'{res_supported_str}'
            res_not_supported_str = ",".join(res_not_supported)
            current_gpu_info['res_not_supported_str'] = f'{res_not_supported_str}'
            
            gpu_info.append({                
                "gpu_i": current_gpu_info.get("res_gpu_i", "0"),
                "current_uuid": current_gpu_info.get("res_current_uuid", "0"),
                "gpu_util": current_gpu_info.get("res_gpu_util", "0"),
                "mem_util": current_gpu_info.get("res_mem_util", "0"),
                "mem_total": current_gpu_info.get("res_mem_total", "0"),
                "mem_used": current_gpu_info.get("res_mem_used", "0"),
                "mem_free": current_gpu_info.get("res_mem_free", "0"),
                "temperature": current_gpu_info.get("res_temperature", "0"),
                "fan_speed": current_gpu_info.get("res_fan_speed", "0"),
                "power_usage": current_gpu_info.get("res_power_usage", "0"),
                "clock_info_graphics": current_gpu_info.get("res_clock_info_graphics", "0"),
                "clock_info_mem": current_gpu_info.get("res_clock_info_mem", "0"),
                "cuda_cores": current_gpu_info.get("res_cuda_cores", "0"),
                "compute_capability": current_gpu_info.get("res_compute_capability", "0"),
                "supported": current_gpu_info.get("res_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0"),
            })
                        
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'

current_gpu_info = get_gpu_info()

async def redis_timer_gpu():
    while True:
        try:
            current_gpu_info = get_gpu_info()
            res_db_gpu = await r.get('db_gpu')
            if res_db_gpu is not None:
                db_gpu = json.loads(res_db_gpu)
                updated_gpu_data = []
                for gpu_i in range(0,len(current_gpu_info)):
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(current_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            else:
                updated_gpu_data = []
                for gpu_i in range(0,len(current_gpu_info)):
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(current_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_timer_gpu())
    asyncio.create_task(redis_timer_network())
    yield

app = FastAPI(lifespan=lifespan)
client = docker.from_env()
device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])

@app.get("/")
async def root():
    return f'Hello from server {os.getenv("BACKEND_PORT")}!'

@app.post("/dockerrest")
async def docker_rest(request: Request):
    try:
        req_data = await request.json()
        print(f' *** got request!')
        print(f' *** req_data')
        print(f' *** {req_data}')

        logging.info(f' *** got request!')
        logging.info(f' *** req_data')
        logging.info(f' *** {req_data}')

        if req_data["req_method"] == "create":
            try:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 1 >>>>>>>>>>>')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 1  >>>>>>>>>>> ')
                
                
                container_name = str(req_data["model_id"]).replace('/', '_')
                container_name = f'vllm_{container_name}'
                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 2 >>>>>>>>>>>')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 2  >>>>>>>>>>> ')
                
                res_db_gpu = await r.get('db_gpu')
                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 3 >>>>>>>>>>>')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 3  >>>>>>>>>>> ')
                
                
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)         
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 4 >>>>>>>>>>>')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 4  >>>>>>>>>>> ')           

                    print(f'SHOULD RESET MEMORY BUT DOESNT DO YEt -> req 1370 clear or in load direct')
                    # torch.cuda.empty_cache()
                    # torch.cuda.reset_max_memory_allocated()
                    
                    
                    
                    # # check if model already downloaded/downloading
                    # all_used_models = [g["used_models"] for g in db_gpu]
                    # print(f'all_used_models {all_used_models}')
                    # if req_data["model_id"] in all_used_models:
                    #     return JSONResponse({"result": 302, "result_data": "Model already downloaded. Trying to start container ..."})
                    
                    # # check if ports already used
                    # all_used_ports = [db_gpu_entry["used_ports"] for db_gpu_entry in db_gpu]
                    # print(f'all_used_ports {all_used_ports}')
                    # if req_data["port_vllm"] in all_used_ports or req_data["port_model"] in all_used_ports:
                    #     return JSONResponse({"result": 409, "result_data": "Error: Port already in use"})
                    
                    # # check if memory available
                    # current_gpu_info = get_gpu_info()
                    # if current_gpu_info[0]["mem_util"] > 50:
                    #     all_running_models = [g["running_model"] for g in db_gpu]
                    #     print(f'all_running_models {all_running_models}')
                    #     for running_model in all_running_models:
                    #         req_container = client.containers.get(req_data["model_id"])
                    #         req_container.stop()
                        
                    # # wait for containers to stop
                    # for i in range(10):
                    #     current_gpu_info = get_gpu_info()
                    #     if current_gpu_info[0]["mem_util"] <= 80:
                    #         continue
                    #     else:
                    #         if i == 9:
                    #             return JSONResponse({"result": 500, "result_data": "Error: Memory > 80%"})
                    #         else:
                    #             time.sleep(1)
                    
                    # get all used ports
                    all_used_ports = []
                    all_used_models = []
                    
                    all_used_ports += [req_data["port_vllm"],req_data["port_model"]]
                    all_used_models += [req_data["port_model"]]
                    
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 5 >>>>>>>>>>>')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 5  >>>>>>>>>>> ')
                    
                    
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "port_vllm": req_data["port_vllm"],
                        "port_model": req_data["port_model"],
                        "used_ports": str(all_used_ports),
                        "used_models": str(all_used_models)
                    }
                    
                    db_gpu += [add_data]
                    # await r.set('db_gpu', json.dumps(db_gpu))                
                
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 6 >>>>>>>>>>>')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 6  >>>>>>>>>>> ')
                else:
                    
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 7 >>>>>>>>>>>')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 7  >>>>>>>>>>> ')
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "port_vllm": str(req_data["port_vllm"]),
                        "port_model": str(req_data["port_model"]),
                        "used_ports": f'{str(req_data["port_vllm"])},{str(req_data["port_model"])}',
                        "used_models": str(str(req_data["model_id"]))
                    }
                    # await r.set('db_gpu', json.dumps(add_data))



                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 8 >>>>>>>>>>>')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 8  >>>>>>>>>>> ')

                print(f'finding containers to stop to free GPU memory...')
                container_list = client.containers.list(all=True)
                
                
                
                
                print(f'found total containers: {len(container_list)}')
                
                
                
                
                
                
                # docker_container_list = get_docker_container_list()
                # docker_container_list_running = [c for c in docker_container_list if c["State"]["Status"] == "running"]
                
                # res_container_list = client.containers.list(all=True)
                # return JSONResponse([container.attrs for container in res_container_list])
                
                # print(f'mhmmhmhmh')
                # vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                # print(f'found total vLLM running containers: {len(vllm_containers_running)}')
                # while len(vllm_containers_running) > 0:
                #     print(f'stopping all vLLM containers...')
                #     for vllm_container in vllm_containers_running:
                #         print(f'stopping container {vllm_container.name}...')
                #         vllm_container.stop()
                #         vllm_container.wait()
                #     print(f'waiting for containers to stop...')
                #     time.sleep(2)
                #     vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                # print(f'all vLLM containers stopped successfully') 
                               
                               
                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 9 >> {req_data["image"]}')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 9  >> {req_data["image"]} ')

                if req_data["image"] == "xoo4foo/zvllm21:latest":
                                     
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 11 >> xoo4foo/zvllm21:latest')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 11  >> xoo4foo/zvllm21:latest ')
                    # if container_name in REDIS DB LISTE ODER QUERY 'list' AM BESTEN MIT self. 
                    container_name = f'{container_name}_{str(int(datetime.now().timestamp()))}'
                    
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [CONTAINER] Starting vLLM container with image: xoo4foo/zvllm21:latest container_name: {container_name}')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [CONTAINER] Starting vLLM container with image: xoo4foo/zvllm21:latest container_name: {container_name}')

                    
                    res_container = client.containers.run(
                        "xoo4foo/zvllm21:latest",
                        command=f'--model {req_data["model_id"]} --tensor-parallel-size {req_data["tensor_parallel_size"]}',
                        name="container_name",
                        runtime="nvidia",
                        volumes={
                            "/models": {"bind": "/models", "mode": "rw"},
                            "../logs": {"bind": "/var/log", "mode": "rw"}
                        },
                        ports={"1370/tcp": ("0.0.0.0", 1370)},
                        shm_size="8gb",
                        environment={
                            "NCCL_DEBUG": "INFO",
                            "VLLM_PORT": "1370"
                        },
                        device_requests=[
                            docker.types.DeviceRequest(
                                count=-1,  # all devices
                                capabilities=[['gpu']]
                            )
                        ],
                        detach=True
                    )
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [CONTAINER] vLLM container started successfully with ID: {res_container.id}')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [CONTAINER] vLLM container started successfully with ID: {res_container.id}')
                
                if req_data["image"] == "vllm/vllm-openai:latest":
                    
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 12 >> vllm/vllm-openai:latest')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 12  >> vllm/vllm-openai:latest ')
                    # if container_name in REDIS DB LISTE ODER QUERY 'list' AM BESTEN MIT self. 
                    container_name = f'{container_name}_{str(int(datetime.now().timestamp()))}'
                                        
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 13 >> container_name {container_name}')
                    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 13  >> container_name {container_name} ')
                    
                    res_container = client.containers.run(
                        req_data["image"],
                        command=f'--model {req_data["model_id"]} --tensor-parallel-size {req_data["tensor_parallel_size"]}',
                        name=container_name,
                        runtime=req_data["runtime"],
                        volumes={
                            "/models": {
                                "bind": "/models",
                                "mode": "rw"
                            }
                        },
                        ports={
                            f'{req_data["port_vllm"]}/tcp': ("0.0.0.0", req_data["port_model"])
                        },
                        ipc_mode="host",
                        device_requests=[device_request],
                        detach=True
                    )
                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 14 >>>>>>>>>>> {res_container}')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 14  >>>>>>>>>>> {res_container}')
                                                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 15 >>>>>>>>>>> {res_container.id}')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 15  >>>>>>>>>>> {res_container.id}')
                                
                # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 9 >>>>>>>>>>> {res_container.id}')
                # logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 9  >>>>>>>>>>> {res_container.id}')
                
                # container_id = res_container.id
                
                
                
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 10 >>>>>>>>>>>')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] create 10  >>>>>>>>>>> ')

                return JSONResponse({"result_status": 200, "result_data": 'whatever'})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                r.delete(f'running_model:{str(req_data["model_id"])}')
                return JSONResponse({"result_status": 404, "result_data": f'{req_data["max_model_len"]}'})
                         
        if req_data["req_method"] == "test":
            print(f'got test!')
            print("req_data")
            print(req_data)
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] test >>>>>>>>>>>')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] test >>>>>>>>>>> req_data["max_model_len"] {req_data["max_model_len"]}')
            
            print("trying request vllm")
            print(req_data["model_id"])
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] test >>>>>>>>>>> trying request vllm ...] {req_data["model_id"]}')
            VLLM_URL = f'http://container_vllm:{os.getenv("VLLM_PORT")}/vllm'
            try:
                response = requests.post(VLLM_URL, json={
                    "req_type":"load",
                    "max_model_len":int(req_data["max_model_len"]),
                    "tensor_parallel_size":int(req_data["tensor_parallel_size"]),
                    "gpu_memory_utilization":float(req_data["gpu_memory_utilization"]),
                    "model":str(req_data["model_id"])
                })
                if response.status_code == 200:
                    logging.info(f' [dockerrest]  status_code: {response.status_code}') 
                    response_json = response.json()
                    logging.info(f' [dockerrest]  response_json: {response_json}') 
                    response_json["result_data"] = response_json["result_data"]
                    return response_json["result_data"]                
                else:
                    logging.info(f' [dockerrest] response: {response}')
                    return JSONResponse({"result_status": 500, "result_data": f'ERRRR'})
            
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'
                                    
        if req_data["req_method"] == "generate":
            print(f'got test!')
            print("req_data")
            print(req_data)
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] generate >>>>>>>>>>>')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [dockerrest] generate >>>>>>>>>>> ')
            
            print("trying request vllm")
            VLLM_URL = f'http://container_vllm:{os.getenv("VLLM_PORT")}/vllm'
            try:
                response = requests.post(VLLM_URL, json={
                    "req_type":"generate",
                    "prompt":req_data["prompt_in"],
                    "temperature":float(req_data["temperature"]),
                    "top_p":float(req_data["top_p"]),
                    "max_tokens":int(req_data["max_tokens"])
                })
                if response.status_code == 200:
                    logging.info(f' [dockerrest]  status_code: {response.status_code}') 
                    response_json = response.json()
                    logging.info(f' [dockerrest]  response_json: {response_json}') 
                    response_json["result_data"] = response_json["result_data"]
                    return response_json["result_data"]                
                else:
                    logging.info(f' [dockerrest] response: {response}')
                    return JSONResponse({"result_status": 500, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
            
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'
           
        if req_data["req_method"] == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            reversed_logs = "\n".join(res_logs_str.splitlines()[::-1])
            return JSONResponse({"result": 200, "result_data": reversed_logs})

        if req_data["req_method"] == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["req_method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["req_method"] == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})


    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{req_data["max_model_len"]}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("BACKEND_IP")}', port=int(os.getenv("BACKEND_PORT")))