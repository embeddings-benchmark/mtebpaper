# MTEB + SGPT


tmp
python beir_dense_retriever.py --modelname /gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit --method weightedmean --dataset trec-covid --specb

<!-- TOC -->

- [MTEB + SGPT](#mteb--sgpt)
    - [Run](#run)
        - [sentence-transformers/sentence-t5-base](#sentence-transformerssentence-t5-base)
    - [Benchmark](#benchmark)
    - [Env Setup](#env-setup)
    - [Model setup](#model-setup)
        - [Python](#python)
            - [Load](#load)
            - [Download](#download)
                - [Data](#data)
                - [Model](#model)
        - [Bash](#bash)
        - [Dataiku](#dataiku)

<!-- /TOC -->

## Run

```python
import os
os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["TRECCOVID"])
evaluation.run(model, output_folder=f"../results/{model_name}")
```

```python
import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-5.8B-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["StackExchangeClustering", "StackExchangeClusteringP2P", "TwentyNewsgroupsClustering", "TwitterSemEval2015", "TwitterURLCorpus", "SciDocs", "StackOverflowDupQuestions"])
evaluation.run(model, output_folder=f"results/{model_name}")
```


### sentence-transformers/sentence-t5-base

```python
import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-5.8B-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["StackExchangeClustering", "StackExchangeClusteringP2P", "TwentyNewsgroupsClustering", "TwitterSemEval2015", "TwitterURLCorpus", "SciDocs", "StackOverflowDupQuestions"])
evaluation.run(model, output_folder=f"results/{model_name}")
```



## Benchmark

Basic
```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model, output_folder=f"results/{model_name}")
```

Online
```python
import os
os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model, output_folder=f"results/{model_name}")
```

Offline
```python
import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model, output_folder=f"results/{model_name}")
```

```python
import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-5.8B-weightedmean-nli-bitfit"
model_name = model_path.split("/")[-1].split("_")[-1]
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["ArxivClusteringP2P"])
evaluation.run(model, output_folder=f"results/{model_name}")
```

## Env Setup

```bash
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n hf-prod python=3.8
conda activate hf-prod

# pt-1.10.1 / cuda 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

!pip install --upgrade git+https://github.com/Muennighoff/mteb.git@offlineaccess
!pip install --upgrade git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings
# If you want to run BEIR tasks
!pip install --upgrade git+https://github.com/beir-cellar/beir.git
```


## Model setup


### Python

#### Load

Load Simple
```python
model = SentenceTransformer("/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit")
```

Load Advanced (Not working)
```python
import os
from sentence_transformers import SentenceTransformer
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/gpfswork/rech/six/commun/models"
cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
model_repo="Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
model = SentenceTransformer(model_repo, cache_folder=cache_folder)
```

#### Download

##### Data



##### Model

Download
```python
import os
import sentence_transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/gpfswork/rech/six/commun/models"
sentence_transformers_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
model_repo="sentence-transformers/sentence-t5-xxl"
revision="0a2f720e57c36306fbfca6025baba48828555764"
model_path = os.path.join(sentence_transformers_cache_dir, model_repo.replace("/", "_"))
model_path_tmp = sentence_transformers.util.snapshot_download(
    repo_id=model_repo,
    revision=revision,
    cache_dir=sentence_transformers_cache_dir,
    library_name="sentence-transformers",
    library_version=sentence_transformers.__version__,
    ignore_files=["flax_model.msgpack", "rust_model.ot", "tf_model.h5",],
)
os.rename(model_path_tmp, model_path)
```


Download (Not working)
```python
import os
os.environ["TRANSFORMERS_CACHE"] = "/gpfswork/rech/six/commun/models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/gpfswork/rech/six/commun/models"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
```

### Bash

```bash
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export SENTENCE_TRANSFORMERS_HOME=$six_ALL_CCFRWORK/models
```


### Dataiku

https://github.com/dataiku/dku-contrib-private/blob/85ae79e9b0a130d954a06ceaf9ca5b99e1029d04/snippets/BUILTIN/python/code_env_resources/variations/sentencetransformer.py

```python
######################## Base imports #################################
import logging
import os
import shutil

from dataiku.code_env_resources import clear_all_env_vars
from dataiku.code_env_resources import grant_permissions
from dataiku.code_env_resources import set_env_path
from dataiku.code_env_resources import set_env_var
from dataiku.code_env_resources import update_models_meta

# Set-up logging
logging.basicConfig()
logger = logging.getLogger("code_env_resources")
logger.setLevel(logging.INFO)

# Clear all environment variables defined by a previously run script
clear_all_env_vars()

######################## Sentence Transformers #################################
# Set sentence_transformers cache directory
set_env_path("SENTENCE_TRANSFORMERS_HOME", "sentence_transformers")

import sentence_transformers

# Download pretrained models
MODELS_REPO_AND_REVISION = [
    ("DataikuNLP/average_word_embeddings_glove.6B.300d", "52d892b217016f53b6c717839bf62c746a658933"),
    ("DataikuNLP/TinyBERT_General_4L_312D", "33ec5b27fcd40369ff402c779baffe219f5360fe"),
    ("DataikuNLP/paraphrase-multilingual-MiniLM-L12-v2", "4f806dbc260d6ce3d6aed0cbf875f668cc1b5480"),
]

sentence_transformers_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
for (model_repo, revision) in MODELS_REPO_AND_REVISION:
    logger.info("Loading pretrained SentenceTransformer model: {}".format(model_repo))
    model_path = os.path.join(sentence_transformers_cache_dir, model_repo.replace("/", "_"))
    
    # Uncomment below to overwrite (force re-download of) all existing models
    # if os.path.exists(model_path):
    #     logger.warning("Removing model: {}".format(model_path))
    #     shutil.rmtree(model_path)

    # This also skips same models with a different revision
    if not os.path.exists(model_path):
        model_path_tmp = sentence_transformers.util.snapshot_download(
            repo_id=model_repo,
            revision=revision,
            cache_dir=sentence_transformers_cache_dir,
            library_name="sentence-transformers",
            library_version=sentence_transformers.__version__,
            ignore_files=["flax_model.msgpack", "rust_model.ot", "tf_model.h5",],
        )
        os.rename(model_path_tmp, model_path)
    else:
        logger.info("Model already downloaded, skipping")
# Add sentence embedding models to the code-envs models meta-data
# (ensure that they are properly displayed in the feature handling)
update_models_meta()
# Grant everyone read access to pretrained models in sentence_transformers/ folder
# (by default, sentence transformers makes them only readable by the owner)
grant_permissions(sentence_transformers_cache_dir)
```
