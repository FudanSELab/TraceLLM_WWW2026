# TraceLLM

This repo contains the code, data, and models for "TraceLLM: Evaluating and Exploring Large Language Models on Distributed System Trace Analysis".

## Our Environment
- Ubuntu 20.04
- Python 3.10.16
- CUDA 12.2
- PyTorch 2.6.0
- NVIDIA 8*A800 GPUs
- 1.0T Memory

## Download
### Step 1: Download Backbone Models
You can download the backbone models according to the following links, and place them in any directory of your choice.
```
# Qwen-3-8B
https://huggingface.co/Qwen/Qwen3-8B

# Llama-3-8B
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# DeepSeek-R1-Distill-Llama-8B
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```
### Step 2: Download Datasets and Outputs
You need to download [**raw_dataset.zip**](https://drive.google.com/file/d/13WtgeJs1cSAkCVXMvmgiDOLAYa79WSJF/view?usp=sharing)[optional], [**TraceBench_demo.zip**](https://drive.google.com/file/d/1nDvjBWi-9JZMSepobacGlikKgspejab6/view?usp=sharing), and [**output.zip**](https://drive.google.com/file/d/1QE8mMArojwCeimeEMPBoxxqhVV-snH4S/view?usp=sharing). Then extract them into **TraceLLM_FT/dataset/raw_dataset**, **TraceLLM_FT/dataset/TraceBench_demo**, and **TraceLLM_FT/output** directories, respectively. The final directory structure should match the following:
```
TraceLLM_FT
├── dataset
│   ├── raw_dataset
│   │   ├── all_traces
│   │   │   ├── all_traces.z01
│   │   │   ├── all_traces.z02
│   │   │   └── ...
│   │   ├── sampled_traces
│   ├── TraceBench_demo
│   │   ├── trace_bench.z01
│   │   ├── trace_bench.z02
│   │   └── ...
├── output
│   ├── DeepSeek-R1-Distill-Llama-8B
│   ├── Meta-Llama-3-8B-Instruct
│   ├── Qwen3-8B
│   ├── graph_llm
│   │   ├── deepseek_model.z01
│   │   ├── deepseek_model.z02
│   │   └── ...
│   ├── LLMresponses_DeepSeek-R1-Distill-Llama-8B_Graph2Token_raw_spans_desc_with_step_lora.json
│   ├── LLMresponses_DeepSeek-R1-Distill-Llama-8B_woFineTuning_woCoT_raw_spans_desc_with_step.json
│   └── ...
├── results
│   ├── benchmark_results
│   │   ├── RQ1
│   │   ├── RQ2
│   │   ├── RQ3
│   ├── TraceLLM_results
│   │   ├── RQ1
│   │   ├── RQ2
├── scripts
│   ├── run_dataset_construction.sh
│   ├── run_test_codePrompt.sh
│   └── ...
├── src
│   ├── ApproachesWithFineTuning
│   ├── ApproachesWithoutFineTuning
│   ├── DatasetConstruction
├── config.yaml
├── README.md
├── requirements.txt
```


## Requirements

```
conda create -n myenv python=3.10.16 -y
conda activate myenv
pip install -r requirements.txt
```


## Dataset Construction
### Construct Dataset from Scratch
```
cd dataset/raw_dataset/all_traces
ls all_traces.z*
7z x all_traces.zip
mv ./all_traces ../all_traces_pre

cd ../../../
chmod +x ./scripts/run_dataset_construction.sh
./scripts/run_dataset_construction.sh
```

TraceBench dataset is stored in `dataset/TraceBench` by default. You can update the default path by modifying `trace_bench_dataset` in our **config.yaml** file. This process is time-consuming, taking about **1 hour**.

### Use the Pre-constructed Dataset (⚠️ **Highly Recommended**)
Since constructing TraceBench can be time-consuming, you can **directly** use the dataset we have constructed. You can find the constructed TraceBench demo dataset in our **dataset/TraceBench_demo** directory.

```
cd dataset/TraceBench_demo
ls trace_bench.z*
7z x trace_bench.zip
mv ./TraceBench_demo ../TraceBench
cd ../../
```


## Baselines - Evaluation
### Step 1: Choose LLM Model
You need to modify our **config.yaml** file, replace `llm_name` with the name of the backbone model to be evaluated, and replace `llm_path` with the corresponding path.
### Step 2: Choose Trace Representation Method
You can modify our **config.yaml** file, update `desc_method` to choose your desired trace representation method, including **adjacency_table_desc**, **edge_list_desc**, **raw_spans_desc**, and **xml_desc**.
### Step 3: Run Evaluation Script
Run the corresponding script to evaluate the trace analysis capabilities of the baseline models.
```
# Raw Instruction
chmod +x ./scripts/run_test_woCoT.sh
./scripts/run_test_woCoT.sh
or
chmod +x ./scripts/run_test_woCoT_api.sh
./scripts/run_test_woCoT_api.sh

# CoT Prompt
chmod +x ./scripts/run_test_wCoT.sh
./scripts/run_test_wCoT.sh

# PoT Prompt
chmod +x ./scripts/run_test_codePrompt.sh
./scripts/run_test_codePrompt.sh
```

You can find our evaluation results of each baseline in our **/output** directory.
```
# Raw Instruction
## Qwen-3-8B
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_woCoT_raw_spans_desc_with_step.json

## Llama-3-8B
output/LLMresponses_Meta-Llama-3-8B-Instruct_woFineTuning_woCoT_raw_spans_desc_with_step.json

## DeepSeek-R1-Distill-Llama-8B
output/LLMresponses_DeepSeek-R1-Distill-Llama-8B_woFineTuning_woCoT_raw_spans_desc_with_step.json

## GPT-4o
output/LLMresponses_GPT4o_woFineTuning_woCoT_raw_spans_desc_without_step_api.json


# CoT Prompt
## Node Sequence
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_w1CoTs_raw_spans_desc_with_step.json

## Adjacency Table-enhanced
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_w1CoTs_adjacency_table_desc_with_step.json

## Edge List-enhanced
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_w1CoTs_edge_list_desc_with_step.json

## Code-like Forms
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_w1CoTs_xml_desc_with_step.json


# PoT Prompt
output/LLMresponses_Qwen3-8B-Instruct_woFineTuning_codePrompt_raw_spans_desc_with_code.json
```


## TraceLLM — Training
### Trace2Text-based Model
#### Step 1: Choose LLM Model
You need to modify our **scripts/run_train_graph2text-model.sh** file, replace `MODEL_NAME` with the name of the backbone model to be fine-tuned, and replace `MODEL_PATH` with the corresponding path.
#### Step 2: Run Tuning Script
Run the corresponding script to fine-tune the baseline models.
```
chmod +x ./scripts/run_train_graph2text-model.sh
./scripts/run_train_graph2text-model.sh
```

You can find our fine-tuned models in our **/output** directory.
```
# Qwen-3-8B
output/Qwen3-8B

# Llama-3-8B
output/Meta-Llama-3-8B-Instruct

# DeepSeek-R1-Distill-Llama-8B
output/DeepSeek-R1-Distill-Llama-8B
```

### Trace2Token-based Model
#### Step 1: Choose LLM Model
You need to modify our **config.yaml** file, replace `llm_name` with the name of the backbone model to be fine-tuned, and replace `llm_path` with the corresponding path.
#### Step 2: Run Tuning Scripts
Run the corresponding script to fine-tune the baseline models.
```
chmod +x ./scripts/run_train_graph2token-model.sh
./scripts/run_train_graph2token-model.sh
```

You can find our fine-tuned models in our **/output** directory.
```
# Qwen-3-8B
## output/graph_llm/qwen_model.z*
cd output/graph_llm
7z x qwen_model.zip

# Llama-3-8B
## output/graph_llm/llama_model.z*
cd output/graph_llm
7z x llama_model.zip 

# DeepSeek-R1-Distill-Llama-8B
## output/graph_llm/deepseek_model.z*
cd output/graph_llm
7z x deepseek_model.zip
```


## TraceLLM - Evaluation
### Trace2Text-based Model
#### Step 1: Choose LLM Model
You need to modify our **scripts/run_test_graph2text-model.sh** file, replace `MODEL_NAME` with the name of the backbone model to be evaluated, and replace `MODEL_PATH` with the corresponding path.
#### Step 2: Run Evaluation Script
Run the corresponding script to evaluate the fine-tuned models.
```
chmod +x ./scripts/run_test_graph2text-model.sh
./scripts/run_test_graph2text-model.sh
```

You can find our evaluation results in our **/output** directory.
```
# Qwen-3-8B
output/Qwen3-8B/lora/sft/raw_spans_desc_with_step/generated_predictions.jsonl

# Llama-3-8B
output/Meta-Llama-3-8B-Instruct/lora/sft/raw_spans_desc_with_step/generated_predictions.jsonl

# DeepSeek-R1-Distill-Llama-8B
output/DeepSeek-R1-Distill-Llama-8B/lora/sft/raw_spans_desc_with_step/generated_predictions.jsonl
```

### Trace2Token-based Model
#### Step 1: Choose LLM Model
You need to modify our **config.yaml** file, replace `llm_name` with the name of the backbone model to be evaluated, and replace `llm_path` with the corresponding path.
#### Step 2: Run Evaluation Script
Run the corresponding script to evaluate the fine-tuned models.
```
chmod +x ./scripts/run_test_graph2token-model.sh
./scripts/run_test_graph2token-model.sh
```

You can find our evaluation results in our **/output** directory.
```
# Qwen-3-8B
output/LLMresponses_Qwen3-8B_Graph2Token_raw_spans_desc_with_step_lora.json

# Llama-3-8B
output/LLMresponses_Meta-Llama-3-8B-Instruct_Graph2Token_raw_spans_desc_with_step_lora.json

# DeepSeek-R1-Distill-Llama-8B
output/LLMresponses_DeepSeek-R1-Distill-Llama-8B_Graph2Token_raw_spans_desc_with_step_lora.json
```

## Experimental Results
You can find all the experimental results in our **/results** directory.