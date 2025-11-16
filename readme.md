# LLMAD: Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection.

## Description
This repository contains the code for the paper: ["Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection"](https://arxiv.org/abs/2405.15370). It demonstrates the use of Large Language Models (LLMs) to tackle the task of Time Series Anomaly Detection.

![LLMAD](assets/method.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Scripts](#running-the-scripts)
    - [Yahoo Dataset](#yahoo-dataset)
    - [WSD Dataset](#wsd-dataset)
    - [KPI Dataset](#kpi-dataset)
- [File Descriptions](#file-descriptions)


## Installation

To get started, clone the repository and install the necessary dependencies:

```shell
# Clone the repository
git clone https://github.com/crepers/LLMAD.git
cd LLMAD

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the necessary dependencies
pip install -r requirements.txt
```

## Usage

### Configuration

Before running the scripts, create a `.env` file in the root of the project by copying the example file:

```shell
cp .env.example .env
```

Then, open the `.env` file and add your API keys and select the model service you want to use:

```env
# Select the model service to use: "openai" or "gemini"
ACTIVE_MODEL_SERVICE="gemini"

# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"python

# Gemini Configuration
GEMINI_API_KEY="your-gemini-api-key"
```

### Running the Scripts

Below are the commands to run the scripts for different datasets.

#### Yahoo Dataset

```shell
bash script/yahoo.sh
```

#### WSD Dataset

```shell
bash script/wsd.sh
```

#### KPI Dataset

```shell
bash script/kpi.sh
```

## File Descriptions

The project is structured with a clear separation of concerns to improve modularity and maintainability.

| File / Folder                 | Description                                                                                              |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| `run.py`                      | The main entry point of the program. Orchestrates the data processing, retrieval, and LLM inference pipeline. |
| `llm_handler.py`              | Handles all interactions with Large Language Models (e.g., OpenAI, Gemini), including API client setup and response parsing. |
| `data_processor.py`           | Responsible for loading, preprocessing, and transforming the time series data before it's fed to the model. |
| `retriever.py`                | Contains functions for retrieving similar time series examples from historical data using DTW.             |
| `Prompt_template.py`          | Defines the structure of the prompts sent to the LLM.                                                    |
| `script/`                     | Contains shell scripts (`yahoo.sh`, `wsd.sh`, `kpi.sh`) for running batch experiments on different datasets. |
| `Eval/`                       | Includes Python scripts for computing evaluation metrics on the model's prediction results.              |
| `run_wsd_interactive.ipynb`   | A Jupyter Notebook for running the WSD experiment step-by-step, allowing for interactive parameter tuning. |
| `visualize_results.ipynb`     | A Jupyter Notebook to visualize the prediction results against the ground truth for easy performance analysis. |
| `.env` / `.env.example`       | Files for managing environment variables, such as API keys and model configurations.                     |
| `requirements.txt`            | Lists all the Python dependencies required for this project.                                             |

If you find this repo helpful, please cite the following papers:
```
@article{liu2024large,
  title={Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection},
  author={Liu, Jun and Zhang, Chaoyun and Qian, Jiaxu and Ma, Minghua and Qin, Si and Bansal, Chetan and Lin, Qingwei and Rajmohan, Saravan and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2405.15370},
  year={2024}
}
```

### References
- https://github.com/lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms?tab=readme-ov-file
- https://github.com/elisejiuqizhang/TS-AD-Datasets?tab=readme-ov-file
- https://github.com/alumik/AnoTransfer-data
