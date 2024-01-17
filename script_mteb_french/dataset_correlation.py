import os
import json
from mteb.abstasks import AbsTask
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_json_files(root_folder):
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


def get_tasks_main_score():
    tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
    tasks_cls = [
        cls(langs=["fr"])
        for cat_cls in tasks_categories_cls
        for cls in cat_cls.__subclasses__()
        if cat_cls.__name__.startswith("AbsTask")
    ]
    tasks_dict = {cls.__class__.__name__: cls for cls in tasks_cls}
    tasks_main_scores = {k: v.description['main_score'] for k, v in tasks_dict.items()}
    return tasks_main_scores


def convert_to_results_dataframe(result_dict, tasks_main_scores, split='test', lang='fr'):
    results_records = []
    subset_main_scores = []
    for model_name, model_results in result_dict.items():
        for dataset_name, dataset_results in model_results.items():
            current_results = dataset_results[split]
            if lang in current_results:
                current_results = current_results[lang]
            if dataset_name in tasks_main_scores:
                main_score = tasks_main_scores[dataset_name]
                subset_main_scores.append((dataset_name, main_score))
                if main_score in current_results:
                    current_results = current_results[main_score]
                else:
                    current_results = None
            else:
                # TODO: improve this clause
                current_results = current_results['cos_sim']['spearman']
                subset_main_scores.append((dataset_name, 'cosine_spearman'))
            results_records.append({'model': model_name, 'dataset': dataset_name, 'result': current_results})
    results_df = pd.DataFrame.from_records(results_records)
    results_pivot = results_df.pivot(index='model', columns='dataset', values='result')
    subset_main_scores = {k: v for k, v in list(set(subset_main_scores))}
    return results_pivot, subset_main_scores


if __name__ == '__main__':
    results_folder_path = '../results'
    result_dict = load_json_files(results_folder_path)
    tasks_main_scores = get_tasks_main_score()
    results_df, subset_main_scores = convert_to_results_dataframe(result_dict, tasks_main_scores, split='test', lang='fr')
    results_df.to_csv('correlation_analysis/results_table.csv')
    # dataset correlations
    spearman_corr_matrix_datasets = results_df.corr(method='spearman')
    spearman_corr_matrix_datasets.to_csv('correlation_analysis/spearman_corr_matrix_datasets.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_datasets, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Dataset Correlation Heatmap (Spearman)')
    plt.savefig('correlation_analysis/spearman_corr_heatmap_datasets.png', bbox_inches='tight')
    with open('correlation_analysis/main_scores.json', 'w') as f:
        json.dump(subset_main_scores, f, indent=4)
    # model correlations
    transposed_results_df = results_df.transpose()
    spearman_corr_matrix_models = transposed_results_df.corr(method='spearman')
    spearman_corr_matrix_models.to_csv('correlation_analysis/spearman_corr_matrix_models.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_models, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Model Correlation Heatmap (Spearman)')
    plt.savefig('correlation_analysis/spearman_corr_heatmap_models.png', bbox_inches='tight')
