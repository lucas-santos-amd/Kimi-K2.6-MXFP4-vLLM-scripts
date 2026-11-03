# -i because worker processes get renamed; match the vllm server, its TP workers,
# and multiprocessing children that carry no recognisable name
pkill -9 -if "vllm|EngineCore|VLLM::|spawn_main|resource_tracker"
until [ "$(rocm-smi --showmemuse | grep -oP 'VRAM%\): \K[0-9]+' | sort -rn | head -1)" -lt 5 ]; do sleep 5; done
rocm-smi --showmemuse | grep VRAM%
