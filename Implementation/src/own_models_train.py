import logging
import os

import jsonplus as json
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import utils
from estimators import OwnEstimator, OwnEstimatorRegression


def get_reduced_column_names():
    return ['mean', 'number_zero_crossings', 'kurtosis', 'skewness', 'hf_diff_acf', 'hf_diff_data',
            'interval_lengths_std', 'sqi_std', 'sqi_min', 'sqi_median', 'peak_range', 'peak_mean',
            'template_corr_highest_sqi_mean', 'template_corr_highest_sqi_std', 'template_corr_median_sqi_mean',
            'template_corr_median_sqi_std', 'interval_means_std', 'sqi_coverage_03', 'sqi_coverage_04',
            'sqi_coverage_05']


def get_final_column_names():
    return ['mean', 'number_zero_crossings', 'kurtosis', 'skewness', 'hf_diff_acf', 'hf_diff_data',
            'interval_lengths_std', 'sqi_std', 'sqi_min', 'sqi_median', 'peak_range', 'peak_mean',
            'template_corr_highest_sqi_mean', 'template_corr_highest_sqi_std', 'template_corr_median_sqi_mean',
            'template_corr_median_sqi_std', 'interval_means_std', 'sqi_coverage_03', 'sqi_coverage_04',
            'sqi_coverage_05', 'hf_ratio_acf', 'hf_ratio_data']


def get_final_models(grid_search, segment_length=10, overlap_amount=0.9, threshold_hr=10):
    paths = get_final_paths(gridsearch=grid_search)
    feature_selection = get_final_column_names()
    recreate_own_models(paths=paths, segment_length=segment_length, overlap_amount=overlap_amount,
                        threshold_hr=threshold_hr, feature_selection=feature_selection, grid_search=grid_search)


def recreate_own_models(paths, segment_length=10, overlap_amount=0.9, threshold_hr=10, grid_search=False,
                        feature_selection=None):
    models = {"rf_clf": RandomForestClassifier(random_state=1, n_jobs=2),
              "rf_regr": RandomForestRegressor(random_state=1, n_jobs=2),
              "xgb_regr": xgb.XGBRegressor(random_state=1, n_jobs=2),
              "xgb_clf": xgb.XGBClassifier(random_state=1, n_jobs=2)}
    if grid_search:
        hyperparameter_paths = ['rf_classificator.json', 'rf_regressor.json', 'xgb_regr.json', 'xgb_clf.json']
        hyperparameter_paths = [os.path.join(utils.get_data_root_path(), 'hyperparameter', path) for path in
                                hyperparameter_paths]
        hyperparameters = []

        for path in hyperparameter_paths:
            with open(path) as file:
                hyperparameters.append(json.loads(file.read()))
    else:
        hyperparameters = [None, None, None, None]
        parameter_paths = ['rf_clf_default.json', 'rf_regr_default.json', 'xgb_default.json', 'xgb_default.json']
        parameter_paths = [os.path.join(utils.get_data_root_path(), 'hyperparameter', path) for path in parameter_paths]
        parameters = []
        for path in parameter_paths:
            with open(path) as file:
                parameters.append(json.loads(file.read()))
        for model, params in zip(models.values(), parameters):
            model.set_params(**params)

    for i, model_key in enumerate(models.keys()):
        if paths[i] in ["RF_Clf_s10_all_h10", "RF_Clf_s10_reduced_h10", "XGB_Clf_s10_all_h10", "RF_Regr_s10_all_h10",
                        "XGB_Regr_s10_all_h10", "RF_Regr_s10_reduced_h10"]:
            continue
        print(paths[i])
        if "clf" in str(model_key):
            OwnEstimator(models[model_key], path=paths[i], feature_selection=feature_selection,
                         segment_length=segment_length, overlap_amount=overlap_amount, hr_threshold=threshold_hr,
                         hyperparameter=hyperparameters[i])
        else:
            OwnEstimatorRegression(models[model_key], path=paths[i], feature_selection=feature_selection,
                                   segment_length=segment_length, overlap_amount=overlap_amount,
                                   hr_threshold=threshold_hr, hyperparameter=hyperparameters[i])


def recreate_reduced_all(grid_search=False, thresholds=[10, 15, 20]):
    feature_selection = get_reduced_column_names()
    for threshold in thresholds:
        logging.info(f"Default segments, all features, threshold={threshold}")
        paths = get_paths(reduced=False, threshold=threshold)
        recreate_own_models(paths=paths, grid_search=grid_search, threshold_hr=threshold)

        logging.info(f"Default segments, reduced feature set, threshold={threshold}")
        paths = get_paths(reduced=True, threshold=threshold)
        recreate_own_models(paths=paths, grid_search=grid_search, feature_selection=feature_selection,
                            threshold_hr=threshold)


def get_paths(reduced=False, segment_length=10, threshold=10):
    paths = ["RF_Clf", "RF_Regr", "XGB_Regr", "XGB_Clf"]
    paths = [path + "_s" + str(segment_length) for path in paths]
    if reduced:
        paths = [path + "_reduced_h" + str(threshold) for path in paths]
    else:
        paths = [path + "_all_h" + str(threshold) for path in paths]
    return paths


def get_final_paths(gridsearch=True, segment_length=10, threshold=10):
    paths = ["RF_Clf", "RF_Regr", "XGB_Regr", "XGB_Clf"]
    paths = [path + "_s" + str(segment_length) for path in paths]
    if not gridsearch:
        paths = [path + "_default_h" + str(threshold) for path in paths]
    else:
        paths = [path + "_h" + str(threshold) for path in paths]
    return paths


def get_default_results(segment_length=10, overlap_amount=0.9, threshold_hr=10):
    feature_selection = get_reduced_column_names()
    paths = ["RF_Clf_default", "RF_Regr_default", "XGB_Regr_default", "XGB_Clf_default"]
    if segment_length != 10 or overlap_amount != 0.9 or threshold_hr != 10:
        paths = [path + "_s" + str(segment_length) for path in paths]
        paths = [path + "_h" + str(threshold_hr) for path in paths]
    recreate_own_models(paths=paths, feature_selection=feature_selection, segment_length=segment_length,
                        overlap_amount=overlap_amount, threshold_hr=threshold_hr, grid_search=False)


def get_default_all_results(segment_length=10, overlap_amount=0.9, threshold_hr=10):
    paths = ["RF_Clf_default_all", "RF_Regr_default_all", "XGB_Regr_default_all", "XGB_Clf_default_all"]
    if segment_length != 10 or overlap_amount != 0.9 or threshold_hr != 10:
        paths = [path + "_s" + str(segment_length) for path in paths]
        paths = [path + "_h" + str(threshold_hr) for path in paths]
    recreate_own_models(paths=paths, segment_length=segment_length, overlap_amount=overlap_amount,
                        threshold_hr=threshold_hr, grid_search=False)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    # get_default_results()
    # get_default_all_results()
    # get_default_results(segment_length=10, overlap_amount=0.9, threshold_hr=5)
    # get_default_all_results(segment_length=10, overlap_amount=0.9, threshold_hr=5)
    # get_default_results(segment_length=10, overlap_amount=0.9, threshold_hr=15)
    # get_default_all_results(segment_length=10, overlap_amount=0.9, threshold_hr=15)
    # try:
    #     get_default_results(segment_length=20, overlap_amount=0.9, threshold_hr=10)
    #     get_default_all_results(segment_length=20, overlap_amount=0.9, threshold_hr=10)
    # except:
    #     logging.error("20 second segments didn't work")
    # try:
    #     get_default_results(segment_length=30, overlap_amount=0.9, threshold_hr=10)
    #     get_default_all_results(segment_length=30, overlap_amount=0.9, threshold_hr=10)
    # except:
    #     logging.error("30 second segments didn't work")
    # try:
    #     get_default_results(segment_length=5, overlap_amount=0.8, threshold_hr=10)
    #     get_default_all_results(segment_length=5, overlap_amount=0.8, threshold_hr=10)
    # except:
    #     logging.error("5 second segments didn't work")
    # recreate_reduced_all(grid_search=True, thresholds=[10, 5, 15])
    get_final_models(grid_search=True)
    pass
