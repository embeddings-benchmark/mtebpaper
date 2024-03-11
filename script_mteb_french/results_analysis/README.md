# MTEB - Results analysis

## This section contains usage information for script relative to results analysis

Before starting, you can create your environment using the packages listed in *requirements.analysis.txt*, by using you favorite environment manager (and using ``pip install -r requirements.analysis.txt``)

### result_parser.py

This scripts intent is to ***format results from json files in the results folder to a table*** (csv, excel or latex).

#### Usage

You can use the class ResultParser like so:

```py
from results_analysis.results_parser import ResultParser

RESULT_FOLDER_PATH = "../results"
rp = ResultParser()
results_df = rp(RESULT_FOLDER_PATH)
results_df.head()
```

You can specify a bunch of parameters. For example,
- apply_style=True will apply styling to the table (bold the max values, center everything, etc)
- output_format="latex" will output a tex file

```py
rp = ResultParser()
results_df = rp(RESULT_FOLDER_PATH, output_format="latex", apply_style=True)
```

Alternatively, you can use a command line :
```
python .\script_mteb_french\results_analysis\results_parser.py --results_folder ./results --output_format csv
```


### data_correlation.py

You can analyse the correlation between you results using this script and the following command : 
```
python .\script_mteb_french\results_analysis\dataset_correlation.py --results_folder ./path/to/results/folder
```

### dataset_similarity.py

This script is used for evaluating the semantic similarity between datasets. Use the following command :
```
python -m script_mteb_french.results_analysis.dataset_similarity --task_type <task-type>

```

### performance_vs_characteristics.py

This script is used for evaluating the correlation between results of a model and its characteristics (input shape, embedding shape, ...). In order to be used, ***you need the caracteristics of the models to be up gathered in a csv file.*** (for an example, look at *model_characteristics.csv*).
```
python ./script_mteb_french/results_analysis/performance_vs_characteristics.py --results_folder ./path/to/results/folder --characteristics_csv ./path/to/csv/file.csv
```

### statistical_tests.py

This script is used to perform statistical tests on the results : it is used to obtain the conover friedman matrix and critical difference diagram.
```
python ./script_mteb_french/results_analysis/statistical_tests.py --results_folder ./path/to/results/folder
```