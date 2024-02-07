import os
import warnings
import json
from argparse import ArgumentParser, Namespace
from mteb.abstasks import AbsTask
import pandas as pd

class ResultsParser:
    """A class to parse the results of MTEB evaluations
    """
    def __init__(self, split:str="test", lang:str="fr") -> None:
        self.split = split
        self.lang = lang

        self.tasks_main_scores_map = ResultsParser.get_tasks_attribute("main_score")
        self.tasks_type_map = ResultsParser.get_tasks_attribute("type")


    def __call__(self, results_folder:str, apply_style:bool=False, save_results:bool=True, return_main_scores:bool=False, **kwargs) -> pd.DataFrame:
        """Wrapper function that crawls through the results folder and build
        a dataframe reprensenting all the results found.

        Args:
            results_folder (str): path to the result folder

        kwargs:
            format (str): format to save the results to. One of excel, csv or latex. Defaults to "excel".

        Returns:
            pd.DataFrame: the results as a dataframe
        """
        result_dict = ResultsParser._load_json_files(results_folder)
        results_df, tasks_main_scores_subset = self._convert_to_results_dataframe(result_dict)
        results_df = self._add_multiindex_to_df(results_df)

        if apply_style:
            results_df = ResultsParser._add_style_to_df(results_df)
        if save_results:
            ResultsParser._save_as_file(results_df, **kwargs)
        if return_main_scores:
            return results_df, tasks_main_scores_subset

        return results_df

        
    @staticmethod
    def _load_json_files(root_folder:str) -> dict:
        """Loads all the json files located in the folder
        (typically containing results) and concatenate them
        in a dict.

        Args:
            root_folder (str): _description_

        Returns:
            dict: the results as a dict
        """
        result_dict = {}

        for root, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    dir_path = os.path.dirname(full_path).replace(root_folder + "/", '')
                    file_name_without_extension = os.path.splitext(file)[0]
                    with open(full_path, 'r') as json_file:
                        json_content = json.load(json_file)

                    if dir_path not in result_dict:
                        result_dict[dir_path] = {}
                    result_dict[dir_path][file_name_without_extension] = json_content

        return result_dict


    @staticmethod
    def get_tasks_attribute(attribute:str='main_score') -> dict[str:str]:
        """Runs through the MTEB module and
        gets the attribute value of each task.

        Args:
            attribute (str): the name of the attribute. Must belong to the
                "description" property of the task.

        Returns:
            dict: a mapping with keys being the tasks names 
                and values being the attributes. Defaults to 'main_score'
        """
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        tasks_cls = [
            cls(langs=["fr"])
            for cat_cls in tasks_categories_cls
            for cls in cat_cls.__subclasses__()
            if cat_cls.__name__.startswith("AbsTask")
        ]
        tasks_dict = {cls.description["name"]: cls for cls in tasks_cls}
        tasks_attribute = {k: v.description[attribute] for k, v in tasks_dict.items()}

        return tasks_attribute


    def _get_task_score(self, task_name:str, task_type:str, task_results:str) -> tuple[float, tuple[str, str]]:
        """Considering a task, gets its results

        Args:
            task_name (str): the name of the task. e.g. "SickFr"
            task_type (str): the type of task. e.g "STS"
            task_results (str): the results of that task. e.g the content of the json result file

        Returns:
            result (float): the value of the metric for obtained for that task
            result_name_score (tuple[str, str]): the name of the task and name of the main scoring metric 
                for that task
        """
        match task_type:
            case "BitextMining":
                print("Results of task BitextMining must be treated separately")
                return None, (task_name, None)
            case other:
                result = task_results[self.split]
                if self.lang in result:
                    result = result[self.lang]
                if task_name in self.tasks_main_scores_map:
                    main_score = self.tasks_main_scores_map[task_name]
                    if main_score in result:
                        result = result[main_score]
                        result_name_score = (task_name, main_score)
                    elif main_score == "cosine_spearman":
                        result = result['cos_sim']['spearman']
                        result_name_score = (task_name, "cosine_spearman")
                    elif main_score == "ap":
                        result = result['cos_sim']['ap']
                        result_name_score = (task_name, "cosine_ap")
                    else:
                        result = None
                        result_name_score = (task_name, None)
                else:
                    warnings.warn(f"Task name '{task_name}' not found in MTEB module.")

        return result, result_name_score


    def _convert_to_results_dataframe(self, result_dict:dict):
        """Converts the results from a dict to a dataframe

        Args:
            result_dict (dict): the result dict returned from load_json_files()
            split (str, optional): the split to get the results from. Defaults to 'test'.
            lang (str, optional): the language to get the results from. Defaults to 'fr'.

        Returns:
            pd.DataFrame: the results as a df
        """
        results_records = []
        tasks_main_scores_subset = []
        for model_name, model_results in result_dict.items():
            for task_name, task_results in model_results.items():
                if task_name in self.tasks_type_map:
                    task_type = self.tasks_type_map[task_name]
                    result, result_name_score = self._get_task_score(task_name, task_type, task_results)
                    results_records.append({'model': model_name, 'dataset': task_name, 'result': result})
                    tasks_main_scores_subset.append(result_name_score)
                else:
                    warnings.warn(f"Task name '{task_name}' not found in MTEB module.")
        results_df = pd.DataFrame.from_records(results_records)
        results_pivot = results_df.pivot(index='model', columns='dataset', values='result')
        tasks_main_scores_subset = dict(tasks_main_scores_subset)

        return results_pivot, tasks_main_scores_subset
    

    def _add_multiindex_to_df(self, results_df:pd.DataFrame) -> pd.DataFrame:
        """Adds a multiindex to columns of the results df
        with the task type

        Args:
            results_df (pd.DataFrame): df returned from _convert_to_results_dataframe()

        Returns:
            pd.DataFrame: the df with a secondary column index
        """
        # reorder column by task type
        reordered_col_names = [col for col in self.tasks_type_map if col in results_df.columns]
        results_df = results_df[reordered_col_names]
        # add multiindex
        multiindex_col_names = [(v,k) for k,v in self.tasks_type_map.items() if k in results_df.columns]
        results_df.columns = pd.MultiIndex.from_tuples(multiindex_col_names)

        return results_df
    
    @staticmethod
    def _add_style_to_df(results_df:pd.DataFrame) -> pd.io.formats:
        """Adds style to the results df.
        - centers values
        - bold the max value of each column

        Args:
            result_df (pd.DataFrame): the result df from add_multiindex_to_df()

        Returns:
            pd.io.formats.style.Styler: the styled df
        """
        def highlight_max(s):
            '''
            highlight the maximum in a Series yellow.
            '''
            is_max = s == s.max()
            return ['font-weight: bold' if v else '' for v in is_max]

        style = results_df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        style.set_table_styles({"model": [{"selector": "th", "props": [("text-align", "left")]}]})
        style.set_properties(**{'text-align': 'center'})
        style.apply(highlight_max)

        return style
    
    @staticmethod
    def _save_as_file(results_df:pd.DataFrame, output_format:str="excel"):
        if output_format not in ["excel", "latex", "csv"]:
            raise ValueError(f"'format' argument should be either excel, latex or csv, not {format}")
        
        match output_format:
            case "csv":
                results_df.style.clear()
                results_df.to_csv("results.csv")
            case "excel":
                results_df.to_excel("results.xlsx")
            case "latex":
                results_df.to_latex("results.tex")
        print("Done !")


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument("--output_format", type=str, choices=["excel", "csv", "latex"], default="excel")
    parser.add_argument("--apply_style", type=bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    rp = ResultsParser()
    results_df = rp(**dict(args._get_kwargs()))