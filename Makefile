IMAGE      := unisoccer
REMOTE     := ujihara@solar.arch.cs.kumamoto-u.ac.jp
REMOTE_PORT := 2222
REMOTE_DIR := /user/arch/ujihara/M2_Research_MyUni
SRC        := SoccerNet/
MATCH_DIR  := SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
JSON_PATH  := $(MATCH_DIR)/clip_dataset.json
CKPT_PATH  := checkpoints/pretrained_classification.pth
COMMENTARY_CKPT    := checkpoints/downstream_commentary_all_open.pth
LLM_CKPT           := meta-llama/Meta-Llama-3-8B-Instruct
INSTRUCTION_CONFIG := configs/instruction_explain.json
INSTRUCTION_CSV    := results/instruction_results.csv
BATCH_SIZE := 4
NUM_WORKERS := 0
MAX_SAMPLES := 0
DEVICE     := cuda
GPU        := 1
OUT_CSV    := results/soccernet_results.csv
COMMENTARY_CSV := results/commentary_results.csv
EXTRA_GAME_TIMES ?=
GAME_TIMES       ?=

TRACKING_ZIP     := SoccerNet/tracking/train.zip
TRACKING_CKPT    := checkpoints/tracking_finetuned.pth
TRACKING_INF_CSV := results/tracking_inference.csv
CAPTION_DIR      ?= SoccerNet/caption-2023/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
TRACKING_OUT     := tracking_clips_sn
TRACKING_SPLIT   := train

DOCKER_RUN := docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
              -e CUDA_VISIBLE_DEVICES=$(GPU) \
              --shm-size=8g \
              -v $(CURDIR):/workspace \
              -v $(CURDIR)/hf_cache:/root/.cache/huggingface

.PHONY: build run preprocess inference inference_local inference_commentary inference_instruction extract_clips download_tracking_captions download_all_tracking preprocess_sn_tracking preprocess_all_tracking verify_sn_tracking train_tracking inference_tracking clean

build:
	docker build --force-rm -t $(IMAGE) .

run:
	docker run -it --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
	    --shm-size=8g \
	    -v $(CURDIR):/workspace \
	    -v $(CURDIR)/hf_cache:/root/.cache/huggingface \
	    $(IMAGE)

preprocess:
	python SoccerNet_script/create_clip_dataset.py \
	    --match_dir "$(MATCH_DIR)"

inference:
	$(DOCKER_RUN) $(IMAGE) \
	    python inference/inference_soccernet.py \
	        --json_path "$(JSON_PATH)" \
	        --ckpt_path $(CKPT_PATH) \
	        --batch_size $(BATCH_SIZE) \
	        --max_samples $(MAX_SAMPLES) \
	        --out_csv $(OUT_CSV)

inference_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_soccernet.py \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(CKPT_PATH) \
	    --batch_size $(BATCH_SIZE) \
	    --max_samples $(MAX_SAMPLES) \
	    --out_csv $(OUT_CSV)

inference_commentary:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_commentary_soccernet.py \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(COMMENTARY_CSV) \
	    --extra_game_times "$(EXTRA_GAME_TIMES)" \
	    --game_times "$(GAME_TIMES)" \
	    --device $(DEVICE)

inference_instruction:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_instruction_soccernet.py \
	    --config $(INSTRUCTION_CONFIG) \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(INSTRUCTION_CSV) \
	    --extra_game_times "$(EXTRA_GAME_TIMES)" \
	    --game_times "$(GAME_TIMES)" \
	    --device $(DEVICE)

download_tracking_captions:
	python SoccerNet_script/download_tracking_captions.py \
	    --tracking_zip "$(TRACKING_ZIP)" \
	    --local_dir SoccerNet \
	    --split $(TRACKING_SPLIT)

download_all_tracking:
	python SoccerNet_script/download_all_tracking.py \
	    --local_dir SoccerNet

preprocess_all_tracking:
	rm -f $(TRACKING_OUT)/soccernet_clips.json
	for split in train valid test; do \
	    if [ -f SoccerNet/tracking/$$split.zip ]; then \
	        python tracking/preprocess/create_soccernet_clips.py \
	            --tracking_zip SoccerNet/tracking/$$split.zip \
	            --caption_dir SoccerNet/caption-2023/ \
	            --out_dir $(TRACKING_OUT) \
	            --split $$split; \
	    else \
	        echo "[skip] SoccerNet/tracking/$$split.zip not found"; \
	    fi; \
	done

preprocess_sn_tracking:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/preprocess/create_soccernet_clips.py \
	    --tracking_zip "$(TRACKING_ZIP)" \
	    --caption_dir "$(CAPTION_DIR)" \
	    --out_dir $(TRACKING_OUT) \
	    --split $(TRACKING_SPLIT)

verify_sn_tracking:
	python -c "\
import json, numpy as np; \
data = json.load(open('$(TRACKING_OUT)/soccernet_clips.json')); \
print(f'ペア数: {len(data)}'); \
e = data[0] if data else None; \
print(f'  seq_id:  {e[\"seq_id\"]}') if e else print('  (no data)'); \
print(f'  caption: {e[\"caption\"][:80]}') if e else None; \
print(f'  shape:   {np.load(e[\"npy_path\"]).shape}') if e else None; \
"

train_tracking:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_tracking.py \
	    --json_path $(TRACKING_OUT)/soccernet_clips.json \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(TRACKING_CKPT) \
	    --device $(DEVICE)

inference_tracking:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_tracking.py \
	    --json_path checkpoints/tracking_test_split.json \
	    --ckpt_path $(TRACKING_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(TRACKING_INF_CSV) \
	    --device $(DEVICE)

extract_clips:
	python SoccerNet_script/extract_clips.py \
	    --results_csv $(COMMENTARY_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --out_dir results/presentation

upload:
	COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -cf - "$(SRC)" | \
	    ssh -p $(REMOTE_PORT) $(REMOTE) \
	        "mkdir -p '$(REMOTE_DIR)' && cd '$(REMOTE_DIR)' && tar -xf -"

clean:
	docker image prune -f
	docker builder prune -f
