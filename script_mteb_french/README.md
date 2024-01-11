# Scripts to run the French MTEB benchmark

This folder contains the scripts used to generate the French tab results on the [MTEB](https://github.com/embeddings-benchmark/mteb) benchmark.

Below are instructions to run the main scripts.

## Benchmark

### Running on host using venv

* Navigate to the repository root folder
* Create your virtual env:

```bash
python3 -m venv .venv
```
* Activate it and install the requirements:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
* Run the benchmark:
```bash
cd script_mteb_french
python run_benchmark.py
```

## Running using Docker

* Navigate to the repository root folder
* Build the docker image:
```bash
docker build -t mtebscripts_image .
```
* Run the benchmark in the container as follows:
```
docker run -v $(pwd):/mtebscripts mtebscripts_image sh -c "cd script_mteb_french && python run_benchmark.py"
```
If you want to use the gpu, make sure to add the `--gpus` option to your run command, or `--runtime=nvidia` if you are using an older version of docker.

Note: Because the volume is shared between the host and the container, the results will be available in the host at the end.

## Models' characteristics

Additionnaly, you can find a script `get_model_specs.py` to compute models' characteristics (size, number of params, embeddings dimension). You can run it similarly to the benchmark by substituting `run_benchmark.py` with `get_model_specs.py`.