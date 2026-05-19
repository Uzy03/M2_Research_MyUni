SHELL      := /bin/bash
.SHELLFLAGS := -o pipefail -c

IMAGE          := unisoccer
CONTAINER_NAME := unisoccer_server
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
GPU        := 0
OUT_CSV    := results/soccernet_results.csv
COMMENTARY_CSV := results/commentary_results.csv
EXTRA_GAME_TIMES ?=
GAME_TIMES       ?=

TRACKING_ZIP     := SoccerNet/tracking/train.zip
TRACKING_CKPT    := checkpoints/tracking_finetuned.pth
TRACKING_INF_CSV := results/tracking_inference.csv

TRAJECTORY_CKPT    := checkpoints/trajectory.pth
TRAJECTORY_CSV     := results/trajectory_inference.csv
TRAJECTORY_K       := 5
TRAJECTORY_CONTEXT := 100
TRAJECTORY_MAXLEN  := 576

REGRESSION_CSV  := results/trajectory_regression_inference.csv

QA_CONFIG       ?= configs/qa_action.json

# Experiment run directory (timestamped)
RUN_TS     ?= $(shell date +%Y%m%d%H%M)
RUN_DIR    ?= checkpoints/$(RUN_TS)
PHASE1_DIR       = $(RUN_DIR)/phase1
# Phase2タグ: init重み(1 or 15) × 指示多様化(0 or 1) でディレクトリを区別
USE_LINEAR            ?= 0
PHASE2_TAG            = init$(if $(filter 1,$(USE_PHASE1_5)),15,1)_div$(INSTRUCTION_DIVERSE)_hub$(if $(filter 1,$(USE_LINEAR)),linear,qformer)$(if $(filter 1,$(USE_SPATIAL)),_spatial,)
SHARED_PHASE1_DIR     = checkpoints/phase1
SHARED_PHASE1_CKPT    = $(SHARED_PHASE1_DIR)/trajectory_regression.pth
SHARED_PHASE1_5_DIR   ?= checkpoints/phase1_5
SHARED_PHASE1_5_CKPT  = $(SHARED_PHASE1_5_DIR)/encoder_contrastive.pth
SHARED_PHASE2_DIR     = checkpoints/phase2_$(PHASE2_TAG)
SHARED_PHASE2_CKPT    = $(SHARED_PHASE2_DIR)/action_alignment.pth
SHARED_PHASE2_5_DIR   = checkpoints/phase2_5_$(PHASE2_TAG)
SHARED_PHASE2_5_CKPT  = $(SHARED_PHASE2_5_DIR)/action_alignment.pth
USE_PHASE1_5          ?= 0
USE_PHASE2_5          ?= 0
INFERENCE_CKPT         = $(if $(filter 1,$(USE_PHASE2_5)),$(SHARED_PHASE2_5_CKPT),$(SHARED_PHASE2_CKPT))
PHASE2_INIT_CKPT      = $(if $(filter 1,$(USE_PHASE1_5)),$(SHARED_PHASE1_5_CKPT),$(SHARED_PHASE1_CKPT))
# アブレーション用 (RUN_TS ベース)
PHASE2_DIR       = $(RUN_DIR)/phase2_$(PHASE2_TAG)
PHASE2_5_DIR     = $(RUN_DIR)/phase2_5_$(PHASE2_TAG)
PHASE3_DIR       = $(RUN_DIR)/phase3_$(PHASE2_TAG)
PHASE4_ALL_DIR   = $(RUN_DIR)/phase4_$(PHASE2_TAG)
PHASE4_DIR       = $(PHASE4_ALL_DIR)/$(basename $(notdir $(QA_CONFIG)))
REGRESSION_CKPT  = $(PHASE1_DIR)/trajectory_regression.pth
ACTION_CKPT      = $(PHASE2_DIR)/action_alignment.pth
PHASE2_5_CKPT    = $(PHASE2_5_DIR)/action_alignment.pth
QA_CSV           = $(PHASE3_DIR)/$(basename $(notdir $(QA_CONFIG)))_results.csv
MAX_GAMES       ?= 0
USE_SPATIAL     ?= 0
LAMBDA_SPATIAL  ?= 0.1
SAVE_INTERVAL   ?= 10
OPEN_LORA       ?= 0
EVAL_INTERVAL   ?= 5
REP_PENALTY     ?= 1.3
MAX_NEW_TOKENS  ?= 40
NUM_BEAMS       ?= 5
LORA_RANK       ?= 32
USE_ANS_TOKEN   ?= 0
QFORMER_HEADS   ?= 1
USE_CHAT_TEMPLATE ?= 0
SHORT_INSTRUCTION ?= 0
CURRICULUM_EPOCHS ?= 5,5,5,5
ALLOWED_TASKS     ?= action
NO_INSTRUCTION    ?= 0
SENTENCE_FORMAT       ?= 0
INSTRUCTION_DIVERSE   ?= 0
ANSWER_DIVERSE        ?= 0
LAMBDA_ALIGN          ?= 0
LAMBDA_SLOT           ?= 0
OPEN_ENCODER          ?= 0
LR_ENCODER            ?= 1e-5
EPOCHS_PHASE1_5       ?= 20
WINDOW_SIZE           ?= 2
TEMPERATURE           ?= 0.07
SOCCERDATA_DIR        ?= /user/arch/ujihara/SoccerData
USE_LLM_QA            ?= 0

INSTRUCTION_ACTION_CKPT := checkpoints/instruction_action.pth
INSTRUCTION_ACTION_CSV  := results/instruction_action_results.csv
SOCCERREPLAY_JSON       := train_data/json/SoccerReplay-1988/classification_train.json
SOCCERREPLAY_VIDEO_BASE ?= /path/to/soccerreplay/videos
HF_TOKEN                ?= $(shell echo $$HF_TOKEN)
CAPTION_DIR      ?= SoccerNet/caption-2023/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
TRACKING_OUT     := tracking_clips_sn
TRACKING_SPLIT   := train

SOCCERDATA_CONFIG := fps1_sec30_onball_step5s
SOCCERDATA_OUT    := soccerdata_clips
SOCCERDATA_MAX_GAMES ?= 0

# SoccerData trajectory training parameters (FPS=1, 30sec clips)
SD_JSON           := $(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG)/clips.json
SD_CKPT           := checkpoints/trajectory_soccerdata.pth
SD_CSV            := results/trajectory_soccerdata_inference.csv
SD_CONTEXT        := 20
SD_K              := 5
SD_STEP           := 1
SD_MAXLEN         := 576
BATCH_PHASE1      ?= 32
BATCH_PHASE2      ?= 4
EPOCHS_PHASE1     ?= 20
EPOCHS_PHASE2     ?= 10
EPOCHS_PHASE2C    ?= 20
EPOCHS_PHASE2_5   ?= 5
CONTRASTIVE_CKPT   = $(PHASE2_DIR)/contrastive.pth
PHASE1_5_DIR       = $(RUN_DIR)/phase1_5
ENCODER_CKPT       = $(PHASE1_5_DIR)/encoder_contrastive.pth

DOCKER_RUN := docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
              -e CUDA_VISIBLE_DEVICES=$(GPU) \
              --shm-size=8g \
              -v $(CURDIR):/workspace \
              -v $(CURDIR)/hf_cache:/root/.cache/huggingface

.PHONY: build start stop exec run preprocess inference inference_local inference_commentary inference_instruction extract_clips \
        download_tracking_captions download_all_tracking \
        preprocess_sn_tracking preprocess_all_tracking _preprocess_all_tracking \
        verify_sn_tracking train_tracking inference_tracking \
        train_instruction inference_instruction_action \
        download_soccerreplay preprocess_soccerdata add_task_labels upload_soccerdata sync_soccerdata \
        compute_spatial_labels \
        train_trajectory_sd train_trajectory_sd_local inference_trajectory_sd inference_trajectory_sd_local \
        train_trajectory train_trajectory_tmux train_trajectory_local inference_trajectory \
        train_trajectory_regression inference_trajectory_regression \
        train_action_alignment run_curriculum inference_soccer_qa \
        run_pipeline run_pipeline_curriculum \
        train_phase1 run_from_phase2 run_curriculum_from_phase2 \
        train_contrastive_phase2 run_contrastive_from_phase2 \
        patch_action_frames train_phase1_5 train_phase1_5_shared run_from_phase1_5 \
        inference_free_qa inference_phase4_all generate_qa_data \
        train_phase2 train_phase2_5 train_phase2_5_shared run_ablation run_inference \
        eval_phase4_judge \
        eval_llm_baseline eval_llm_baseline_judge \
        check smoke smoke_phase2 clean

build:
	docker build --force-rm -t $(IMAGE) .

start:
	docker run -d --name $(CONTAINER_NAME) \
	    --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    --shm-size=8g \
	    -v $(CURDIR):/workspace \
	    -v $(CURDIR)/hf_cache:/root/.cache/huggingface \
	    $(IMAGE)
	@echo "Container started: $(CONTAINER_NAME)"
	@echo "Enter with: make exec"

stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

exec:
	docker exec -it $(CONTAINER_NAME) bash

run:
	docker run -it --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
	    --shm-size=8g \
	    -v $(CURDIR):/workspace \
	    -v $(CURDIR)/hf_cache:/root/.cache/huggingface \
	    $(IMAGE) bash

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
	bash tmux_run.sh download_all_tracking \
	    "python SoccerNet_script/download_all_tracking.py --local_dir SoccerNet"

preprocess_sn_tracking:
	bash tmux_run.sh preprocess_sn_tracking \
	    "python tracking/preprocess/create_soccernet_clips.py \
	        --tracking_zip '$(TRACKING_ZIP)' \
	        --caption_dir '$(CAPTION_DIR)' \
	        --out_dir $(TRACKING_OUT) \
	        --split $(TRACKING_SPLIT)"

preprocess_all_tracking:
	bash tmux_run.sh preprocess_all_tracking "$(MAKE) _preprocess_all_tracking"

_preprocess_all_tracking:
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
	bash tmux_run.sh train_tracking \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_tracking.py \
	        --json_path $(TRACKING_OUT)/soccernet_clips.json \
	        --ckpt_path $(COMMENTARY_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_ckpt $(TRACKING_CKPT) \
	        --device $(DEVICE)"

inference_tracking:
	bash tmux_run.sh inference_tracking \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_tracking.py \
	        --json_path checkpoints/tracking_test_split.json \
	        --ckpt_path $(TRACKING_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_csv $(TRACKING_INF_CSV) \
	        --device $(DEVICE)"

extract_clips:
	python SoccerNet_script/extract_clips.py \
	    --results_csv $(COMMENTARY_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --out_dir results/presentation

upload:
	COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -cf - "$(SRC)" | \
	    ssh -p $(REMOTE_PORT) $(REMOTE) \
	        "mkdir -p '$(REMOTE_DIR)' && cd '$(REMOTE_DIR)' && tar -xf -"

train_trajectory:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory.py \
	    --json_path $(TRACKING_OUT)/soccernet_clips.json \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(TRAJECTORY_CKPT) \
	    --K $(TRAJECTORY_K) \
	    --context_len $(TRAJECTORY_CONTEXT) \
	    --max_length $(TRAJECTORY_MAXLEN) \
	    --device $(DEVICE)

train_trajectory_tmux:
	bash tmux_run.sh train_trajectory \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory.py \
	        --json_path $(TRACKING_OUT)/soccernet_clips.json \
	        --ckpt_path $(COMMENTARY_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_ckpt $(TRAJECTORY_CKPT) \
	        --K $(TRAJECTORY_K) \
	        --context_len $(TRAJECTORY_CONTEXT) \
	        --max_length $(TRAJECTORY_MAXLEN) \
	        --device $(DEVICE)"

train_trajectory_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory.py \
	    --json_path $(TRACKING_OUT)/soccernet_clips.json \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(TRAJECTORY_CKPT) \
	    --K $(TRAJECTORY_K) \
	    --context_len $(TRAJECTORY_CONTEXT) \
	    --max_length $(TRAJECTORY_MAXLEN) \
	    --device $(DEVICE)

inference_trajectory:
	bash tmux_run.sh inference_trajectory \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_trajectory.py \
	        --json_path checkpoints/trajectory_test_split.json \
	        --ckpt_path $(TRAJECTORY_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_csv $(TRAJECTORY_CSV) \
	        --K $(TRAJECTORY_K) \
	        --context_len $(TRAJECTORY_CONTEXT) \
	        --device $(DEVICE)"

inference_trajectory_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_trajectory.py \
	    --json_path checkpoints/trajectory_test_split.json \
	    --ckpt_path $(TRAJECTORY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(TRAJECTORY_CSV) \
	    --K $(TRAJECTORY_K) \
	    --context_len $(TRAJECTORY_CONTEXT) \
	    --device $(DEVICE)

download_soccerreplay:
	python SoccerNet_script/download_soccerreplay.py \
	    --token "$(HF_TOKEN)" \
	    --out_dir train_data/json/SoccerReplay-1988

train_instruction:
	bash tmux_run.sh train_instruction \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python inference/train_instruction.py \
	        --json_path $(SOCCERREPLAY_JSON) \
	        --video_base '$(SOCCERREPLAY_VIDEO_BASE)' \
	        --ckpt_path $(COMMENTARY_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_ckpt $(INSTRUCTION_ACTION_CKPT) \
	        --device $(DEVICE)"

inference_instruction_action:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_instruction_soccernet.py \
	    --config configs/instruction_action.json \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(INSTRUCTION_ACTION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(INSTRUCTION_ACTION_CSV) \
	    --device $(DEVICE)

train_trajectory_sd:
	bash tmux_run.sh train_trajectory_sd \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory.py \
	        --json_path $(SD_JSON) \
	        --ckpt_path $(COMMENTARY_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_ckpt $(SD_CKPT) \
	        --context_len $(SD_CONTEXT) \
	        --K $(SD_K) \
	        --step $(SD_STEP) \
	        --max_length $(SD_MAXLEN) \
	        --device $(DEVICE)"

train_trajectory_sd_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(SD_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --K $(SD_K) \
	    --step $(SD_STEP) \
	    --max_length $(SD_MAXLEN) \
	    --device $(DEVICE)

inference_trajectory_sd:
	bash tmux_run.sh inference_trajectory_sd \
	    "CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_trajectory.py \
	        --json_path checkpoints/trajectory_soccerdata_test_split.json \
	        --ckpt_path $(SD_CKPT) \
	        --llm_ckpt $(LLM_CKPT) \
	        --out_csv $(SD_CSV) \
	        --K $(SD_K) \
	        --context_len $(SD_CONTEXT) \
	        --device $(DEVICE)"

inference_trajectory_sd_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_trajectory.py \
	    --json_path checkpoints/trajectory_soccerdata_test_split.json \
	    --ckpt_path $(SD_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(SD_CSV) \
	    --K $(SD_K) \
	    --context_len $(SD_CONTEXT) \
	    --device $(DEVICE)

preprocess_soccerdata:
	python SoccerNet_script/preprocess_soccerdata.py \
	    --data_dir $(SOCCERDATA_DIR) \
	    --out_dir $(SOCCERDATA_OUT) \
	    --config $(SOCCERDATA_CONFIG) \
	    --max_games $(SOCCERDATA_MAX_GAMES)

add_task_labels:
	python SoccerNet_script/add_task_labels.py \
	    --json_path $(SD_JSON)

upload_soccerdata:
	COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -cf - \
	    "$(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG)" | \
	    ssh -p $(REMOTE_PORT) $(REMOTE) \
	        "mkdir -p '$(REMOTE_DIR)/soccerdata_clips' && \
	         cd '$(REMOTE_DIR)' && tar -xf -"

sync_soccerdata:
	rsync -avz --progress --partial --ignore-existing \
	    --exclude='._*' --exclude='.DS_Store' \
	    -e "ssh -o StrictHostKeyChecking=no -p $(REMOTE_PORT)" \
	    "$(SOCCERDATA_DIR)/" \
	    "$(REMOTE):/user/arch/ujihara/SoccerData/"

# ルールベース空間ラベル計算（formation + defensive line height）
# 使い方: make compute_spatial_labels
compute_spatial_labels:
	python tracking/compute_spatial_labels.py \
	    --clips_json $(SD_JSON) \
	    --base_dir $(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG) \
	    --out_json $(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG)/spatial_labels.json \
	    --max_games $(MAX_GAMES) \
	    --save_interval $(SAVE_INTERVAL)

train_trajectory_regression:
	mkdir -p $(PHASE1_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory_regression.py \
	    --json_path $(SD_JSON) \
	    --out_ckpt $(REGRESSION_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --K $(SD_K) \
	    --step $(SD_STEP) \
	    --batch_size $(BATCH_PHASE1) \
	    --epochs $(EPOCHS_PHASE1) \
	    --max_games $(MAX_GAMES) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE1_DIR)/train.log

train_phase1:
	$(MAKE) train_trajectory_regression \
	    PHASE1_DIR=$(SHARED_PHASE1_DIR) \
	    REGRESSION_CKPT=$(SHARED_PHASE1_CKPT)

train_phase1_5_shared:
	$(MAKE) train_phase1_5 \
	    PHASE1_5_DIR=$(SHARED_PHASE1_5_DIR) \
	    ENCODER_CKPT=$(SHARED_PHASE1_5_CKPT)

inference_trajectory_regression:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_trajectory_regression.py \
	    --json_path $(PHASE1_DIR)/trajectory_regression_test_split.json \
	    --ckpt_path $(REGRESSION_CKPT) \
	    --out_csv $(REGRESSION_CSV) \
	    --K $(SD_K) \
	    --context_len $(SD_CONTEXT) \
	    --device $(DEVICE)

train_action_alignment:
	mkdir -p $(PHASE2_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(REGRESSION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(ACTION_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --epochs $(EPOCHS_PHASE2) \
	    --max_games $(MAX_GAMES) \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    $(if $(ALLOWED_TASKS),--allowed_tasks $(ALLOWED_TASKS),) \
	    $(if $(filter 1,$(NO_INSTRUCTION)),--no_instruction,) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    $(if $(filter 1,$(INSTRUCTION_DIVERSE)),--instruction_diverse,) \
	    $(if $(filter 1,$(ANSWER_DIVERSE)),--answer_diverse,) \
	    $(if $(filter-out 0,$(LAMBDA_ALIGN)),--lambda_align $(LAMBDA_ALIGN),) \
	    $(if $(filter-out 0,$(LAMBDA_SLOT)),--lambda_slot $(LAMBDA_SLOT),) \
	    $(if $(filter 1,$(OPEN_ENCODER)),--open_visual_encoder --lr_encoder $(LR_ENCODER),) \
	    $(if $(filter 1,$(USE_LLM_QA)),--use_llm_qa,) \
	    $(if $(filter 1,$(USE_LINEAR)),--use_linear,) \
	    $(if $(filter 1,$(USE_SPATIAL)),--spatial_labels $(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG)/spatial_labels.json,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_DIR)/train.log

run_curriculum:
	mkdir -p $(PHASE2_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(REGRESSION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(ACTION_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --max_games $(MAX_GAMES) \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    --curriculum_stages $(CURRICULUM_EPOCHS) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_DIR)/curriculum.log

test_phase2:
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(ACTION_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --max_games $(MAX_GAMES) \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    $(if $(filter 1,$(INSTRUCTION_DIVERSE)),--instruction_diverse,) \
	    $(if $(filter-out 0,$(LAMBDA_SLOT)),--lambda_slot $(LAMBDA_SLOT),) \
	    $(if $(ALLOWED_TASKS),--allowed_tasks $(ALLOWED_TASKS),) \
	    --test_only \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_DIR)/test.log

inference_soccer_qa:
	mkdir -p $(PHASE3_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_soccer_qa.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(ACTION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(PHASE3_DIR)/results.csv \
	    --context_len $(SD_CONTEXT) \
	    --max_games $(MAX_GAMES) \
	    --repetition_penalty $(REP_PENALTY) \
	    --max_new_tokens $(MAX_NEW_TOKENS) \
	    --num_beams $(NUM_BEAMS) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    $(if $(ALLOWED_TASKS),--tasks $(ALLOWED_TASKS),) \
	    $(if $(QA_CONFIG),--free_config $(QA_CONFIG),) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE3_DIR)/inference.log

inference_free_qa:
	mkdir -p $(PHASE4_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_soccer_qa.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(ACTION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(PHASE4_DIR)/results.csv \
	    --context_len $(SD_CONTEXT) \
	    --max_games $(MAX_GAMES) \
	    --repetition_penalty $(REP_PENALTY) \
	    --max_new_tokens $(MAX_NEW_TOKENS) \
	    --num_beams $(NUM_BEAMS) \
	    --qformer_heads $(QFORMER_HEADS) \
	    --tasks none \
	    $(if $(QA_CONFIG),--free_config $(QA_CONFIG),) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE4_DIR)/inference.log

inference_phase4_all:
	mkdir -p $(PHASE4_ALL_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_soccer_qa.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(ACTION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(PHASE4_ALL_DIR)/results.csv \
	    --context_len $(SD_CONTEXT) \
	    --max_games $(MAX_GAMES) \
	    --repetition_penalty $(REP_PENALTY) \
	    --max_new_tokens $(MAX_NEW_TOKENS) \
	    --num_beams $(NUM_BEAMS) \
	    --qformer_heads $(QFORMER_HEADS) \
	    --tasks none \
	    --free_configs configs/qa_formation.json configs/qa_commentary.json configs/qa_attacking_intent.json configs/qa_defensive_intent.json configs/qa_defensive_line.json \
	    --phase4_base_dir $(PHASE4_ALL_DIR) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE4_ALL_DIR)/inference.log

run_pipeline:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) train_trajectory_regression RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) train_action_alignment RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) inference_soccer_qa RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)

run_pipeline_curriculum:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) train_trajectory_regression RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) run_curriculum RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) inference_soccer_qa RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)

run_from_phase2:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) train_action_alignment \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) \
	    REGRESSION_CKPT=$(PHASE2_INIT_CKPT)
	$(MAKE) inference_soccer_qa RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) inference_phase4_all RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT) GPU=$(GPU)

run_curriculum_from_phase2:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) run_curriculum \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) \
	    REGRESSION_CKPT=$(PHASE2_INIT_CKPT)
	$(MAKE) inference_soccer_qa RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) inference_phase4_all RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT) GPU=$(GPU)

train_contrastive_phase2:
	mkdir -p $(PHASE2_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) \
	python tracking/train_contrastive_phase2.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(SHARED_PHASE1_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(CONTRASTIVE_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --epochs $(EPOCHS_PHASE2C) \
	    --max_games $(MAX_GAMES) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_DIR)/train.log

run_contrastive_from_phase2:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) train_contrastive_phase2 RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES)
	$(MAKE) inference_soccer_qa RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) \
	    ACTION_CKPT=$(CONTRASTIVE_CKPT)
	$(MAKE) inference_phase4_all RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT) GPU=$(GPU)

patch_action_frames:
	python SoccerNet_script/patch_action_frames.py \
	    --json_path $(SD_JSON) \
	    --data_dir $(SOCCERDATA_DIR) \
	    --max_games $(SOCCERDATA_MAX_GAMES)

generate_qa_data:
	CUDA_VISIBLE_DEVICES=$(GPU) python SoccerNet_script/generate_qa_data.py \
	    --json_path $(SD_JSON) \
	    --model meta-llama/Meta-Llama-3-8B-Instruct \
	    --max_games $(MAX_GAMES) \
	    --save_interval 10

train_phase1_5:
	mkdir -p $(PHASE1_5_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) \
	python tracking/train_contrastive_phase2.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(SHARED_PHASE1_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(ENCODER_CKPT) \
	    --batch_size $(BATCH_PHASE2) \
	    --epochs $(EPOCHS_PHASE1_5) \
	    --max_games $(MAX_GAMES) \
	    --window_size $(WINDOW_SIZE) \
	    --temperature $(TEMPERATURE) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE1_5_DIR)/train.log

# Phase 2 共有チェックポイント保存 (phase1/phase1.5 と同じイメージ)
# 使い方: make train_phase2 USE_PHASE1_5=1 INSTRUCTION_DIVERSE=1 GPU=0
train_phase2:
	$(MAKE) train_action_alignment \
	    PHASE2_DIR=$(SHARED_PHASE2_DIR) \
	    ACTION_CKPT=$(SHARED_PHASE2_CKPT) \
	    REGRESSION_CKPT=$(PHASE2_INIT_CKPT)

train_phase2_5:
	mkdir -p $(PHASE2_5_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(SHARED_PHASE2_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(PHASE2_5_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --epochs $(EPOCHS_PHASE2_5) \
	    --max_games $(MAX_GAMES) \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    $(if $(ALLOWED_TASKS),--allowed_tasks $(ALLOWED_TASKS),) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    $(if $(filter 1,$(INSTRUCTION_DIVERSE)),--instruction_diverse,) \
	    --use_llm_qa \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_5_DIR)/train.log

# Phase 2.5 共有チェックポイント保存 (phase2 と同じイメージ)
# 使い方: make train_phase2_5_shared USE_LINEAR=0 INSTRUCTION_DIVERSE=1 SENTENCE_FORMAT=1 GPU=0
train_phase2_5_shared:
	$(MAKE) train_phase2_5 \
	    PHASE2_5_DIR=$(SHARED_PHASE2_5_DIR) \
	    PHASE2_5_CKPT=$(SHARED_PHASE2_5_CKPT)

# Phase 2.5.1: 既存 Phase 2.5 重みから spatial タスクを追加して継続学習
# 使い方: make train_phase2_5_1 USE_LINEAR=1 INSTRUCTION_DIVERSE=1 SENTENCE_FORMAT=1 GPU=0 MAX_GAMES=5
PHASE2_5_TAG_NO_SPATIAL = init$(if $(filter 1,$(USE_PHASE1_5)),15,1)_div$(INSTRUCTION_DIVERSE)_hub$(if $(filter 1,$(USE_LINEAR)),linear,qformer)
PHASE2_5_1_DIR = checkpoints/phase2_5_1_$(PHASE2_5_TAG_NO_SPATIAL)
train_phase2_5_1:
	mkdir -p $(PHASE2_5_1_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path checkpoints/phase2_5_$(PHASE2_5_TAG_NO_SPATIAL)/action_alignment.pth \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(PHASE2_5_1_DIR)/action_alignment.pth \
	    --context_len $(SD_CONTEXT) \
	    --batch_size $(BATCH_PHASE2) \
	    --epochs $(EPOCHS_PHASE2_5) \
	    --max_games $(MAX_GAMES) \
	    $(if $(filter 1,$(USE_LINEAR)),--use_linear,) \
	    $(if $(filter 1,$(SENTENCE_FORMAT)),--sentence_format,) \
	    $(if $(filter 1,$(INSTRUCTION_DIVERSE)),--instruction_diverse,) \
	    --allowed_tasks action,formation,def_line \
	    --use_llm_qa \
	    --spatial_labels $(SOCCERDATA_OUT)/$(SOCCERDATA_CONFIG)/spatial_labels.json \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_5_1_DIR)/train.log

# アブレーション: Phase 2.5 以降を一括実行 (Phase 2 は train_phase2 で事前完了が前提)
# 使い方: make run_ablation USE_PHASE1_5=1 INSTRUCTION_DIVERSE=1 GPU=0 MAX_GAMES=5
run_ablation:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	@echo "=== Ablation Phase2.5+: $(PHASE2_TAG) (Phase2 ckpt: $(SHARED_PHASE2_CKPT)) ==="
	$(MAKE) train_phase2_5 \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) GPU=$(GPU) \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT) INSTRUCTION_DIVERSE=$(INSTRUCTION_DIVERSE)
	$(MAKE) inference_soccer_qa \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) GPU=$(GPU) \
	    ACTION_CKPT=$(PHASE2_5_CKPT) \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT)
	$(MAKE) inference_phase4_all \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) GPU=$(GPU) \
	    ACTION_CKPT=$(PHASE2_5_CKPT) \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT)

# Phase 3 + Phase 4 のみ実行 (USE_PHASE2_5=0: Phase2 ckpt, USE_PHASE2_5=1: Phase2.5 ckpt)
# 使い方: make run_inference USE_PHASE1_5=0 INSTRUCTION_DIVERSE=1 USE_LINEAR=0 GPU=0
#         make run_inference USE_PHASE2_5=1 USE_LINEAR=0 INSTRUCTION_DIVERSE=1 GPU=0
run_inference:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	@echo "=== Inference only: $(PHASE2_TAG) (ckpt: $(INFERENCE_CKPT)) ==="
	$(MAKE) inference_soccer_qa \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) GPU=$(GPU) \
	    ACTION_CKPT=$(INFERENCE_CKPT) \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT)
	$(MAKE) inference_phase4_all \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) GPU=$(GPU) \
	    ACTION_CKPT=$(INFERENCE_CKPT) \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT)

run_from_phase1_5:
	$(eval RUN_TS := $(shell date +%Y%m%d%H%M))
	$(MAKE) train_phase1_5 \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT)
	$(MAKE) train_action_alignment \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) \
	    REGRESSION_CKPT=checkpoints/$(RUN_TS)/phase1_5/encoder_contrastive.pth \
	    SENTENCE_FORMAT=$(SENTENCE_FORMAT) INSTRUCTION_DIVERSE=$(INSTRUCTION_DIVERSE) \
	    LAMBDA_SLOT=$(LAMBDA_SLOT) EPOCHS_PHASE2=$(EPOCHS_PHASE2)
	$(MAKE) inference_soccer_qa \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT)
	$(MAKE) inference_phase4_all \
	    RUN_TS=$(RUN_TS) MAX_GAMES=$(MAX_GAMES) SENTENCE_FORMAT=$(SENTENCE_FORMAT) GPU=$(GPU)

check:
	python -m py_compile tracking/train_action_alignment.py
	python -m py_compile tracking/train_trajectory_regression.py
	python -m py_compile tracking/train_contrastive_phase2.py
	python -m py_compile tracking/dataset/multitask_dataset.py
	python -m py_compile tracking/dataset/window_dataset.py
	python -m py_compile tracking/encoder.py
	python -m py_compile SoccerNet_script/add_task_labels.py
	python -m py_compile SoccerNet_script/patch_action_frames.py
	python tracking/tests/test_phase2.py
	@echo "All checks passed!"

smoke:
	$(eval SMOKE_TS := $(shell date +%Y%m%d%H%M)_smoke)
	$(eval SMOKE_DIR := checkpoints/$(SMOKE_TS))
	$(eval SMOKE_P1  := $(SMOKE_DIR)/phase1)
	$(eval SMOKE_P2  := $(SMOKE_DIR)/phase2)
	$(eval SMOKE_P3  := $(SMOKE_DIR)/phase3)
	mkdir -p $(SMOKE_P1) $(SMOKE_P2) $(SMOKE_P3)
	@echo "=== smoke: Phase 1 ==="
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_trajectory_regression.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path "" \
	    --out_ckpt $(SMOKE_P1)/trajectory_regression.pth \
	    --context_len $(SD_CONTEXT) \
	    --K $(SD_K) --step $(SD_STEP) \
	    --batch_size 2 --epochs 1 \
	    --max_samples 50 \
	    --device $(DEVICE) \
	    2>&1 | tee $(SMOKE_P1)/smoke.log
	@echo "=== smoke: Phase 2 ==="
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(SMOKE_P1)/trajectory_regression.pth \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(SMOKE_P2)/action_alignment.pth \
	    --context_len $(SD_CONTEXT) \
	    --batch_size 2 --epochs 1 \
	    --max_samples 50 \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    --sentence_format --instruction_diverse \
	    --lambda_slot 0.1 \
	    --device $(DEVICE) \
	    2>&1 | tee $(SMOKE_P2)/smoke.log
	@echo "=== smoke: Phase 3 ==="
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/inference_soccer_qa.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(SMOKE_P2)/action_alignment.pth \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(SMOKE_P3)/smoke_results.csv \
	    --context_len $(SD_CONTEXT) \
	    --max_samples 50 \
	    --repetition_penalty $(REP_PENALTY) \
	    --max_new_tokens $(MAX_NEW_TOKENS) \
	    --num_beams $(NUM_BEAMS) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(SMOKE_P3)/smoke.log
	@echo "=== smoke done: $(SMOKE_DIR) ==="

smoke_phase2:
	mkdir -p $(PHASE2_DIR)
	TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$(GPU) python tracking/train_action_alignment.py \
	    --json_path $(SD_JSON) \
	    --ckpt_path $(REGRESSION_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_ckpt $(ACTION_CKPT) \
	    --context_len $(SD_CONTEXT) \
	    --batch_size 2 \
	    --epochs 1 \
	    --max_samples 50 \
	    $(if $(filter 1,$(OPEN_LORA)),--open_lora,) \
	    --lora_rank $(LORA_RANK) \
	    $(if $(filter 1,$(USE_ANS_TOKEN)),--use_ans_token,) \
	    --qformer_heads $(QFORMER_HEADS) \
	    $(if $(filter 1,$(USE_CHAT_TEMPLATE)),--use_chat_template,) \
	    $(if $(filter 1,$(SHORT_INSTRUCTION)),--short_instruction,) \
	    --device $(DEVICE) \
	    2>&1 | tee $(PHASE2_DIR)/smoke.log

# Phase 4 LLM-as-a-Judge 評価
# LLM ベースライン評価（メタデータのみで Phase 4 と同じ質問に回答）
# 使い方: make eval_llm_baseline GPU=0
# CLIP_IDS_FROM: Phase4 results.json のパス（同じクリップで公平比較するため）
# 例: make eval_llm_baseline CLIP_IDS_FROM=checkpoints/202605151624/phase4_init1_div1_hublinear/qa_formation/results.json GPU=0
CLIP_IDS_FROM ?=
PHASE4_CONFIGS ?= qa_formation qa_commentary qa_attacking_intent qa_defensive_intent qa_defensive_line

eval_llm_baseline:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/eval_llm_baseline.py \
	    --clips_json $(SD_JSON) \
	    --config_dir configs \
	    --configs $(PHASE4_CONFIGS) \
	    --out_dir checkpoints/llm_baseline \
	    --llm_ckpt meta-llama/Meta-Llama-3-8B-Instruct \
	    --device $(DEVICE) \
	    $(if $(CLIP_IDS_FROM),--clip_ids_from $(CLIP_IDS_FROM),)

eval_llm_baseline_judge:
	$(MAKE) eval_llm_baseline GPU=$(GPU) CLIP_IDS_FROM=$(CLIP_IDS_FROM)
	$(MAKE) eval_phase4_judge PHASE4_DIR=checkpoints/llm_baseline GPU=$(GPU)

# 使い方: make eval_phase4_judge PHASE4_DIR=checkpoints/RUN_TS/phase4_TAG GPU=0
eval_phase4_judge:
	CUDA_VISIBLE_DEVICES=$(GPU) python tracking/eval_phase4_judge.py \
	    --phase4_dir $(PHASE4_DIR) \
	    --llm_ckpt meta-llama/Meta-Llama-3-8B-Instruct \
	    --configs $(PHASE4_CONFIGS) \
	    --device $(DEVICE)

clean:
	docker image prune -f
	docker builder prune -f
