import logging
import os

import jsonplus as json
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import utils
from estimators import OwnEstimator, OwnEstimatorRegression


def recreate_own_models(paths, segment_length=10, overlap_amount=0.9, threshold_hr=10, grid_search=False,
                        feature_selection=None):
    models = {"rf_clf": RandomForestClassifier(random_state=1, verbose=1),
              "rf_regr": RandomForestRegressor(random_state=1, verbose=1),
              "xgb_regr": xgb.XGBRegressor(random_state=1, verbosity=1),
              "xgb_clf": xgb.XGBClassifier(random_state=1, verbosity=1)}
    if grid_search:
        hyperparameter_paths = ['rf_classificator.json', 'rf_regressor.json', 'xgb.json', 'xgb.json']
        hyperparameter_paths = [os.path.join(utils.get_data_root_path(), 'hyperparameter', path) for path in
                                hyperparameter_paths]
        hyperparameters = []
        for path in hyperparameter_paths:
            with open(path) as file:
                hyperparameters.append(json.loads(file.read()))
    else:
        hyperparameters = [None, None, None, None]

    for i, model_key in enumerate(models.keys()):
        if "clf" in str(model_key):
            model = OwnEstimator(models[model_key], path=paths[i], feature_selection=feature_selection,
                                 segment_length=segment_length, overlap_amount=overlap_amount,
                                 hr_threshold=threshold_hr, hyperparameter=hyperparameters[i])
        else:
            model = OwnEstimatorRegression(models[model_key], path=paths[i], feature_selection=feature_selection,
                                           segment_length=segment_length, overlap_amount=overlap_amount,
                                           hr_threshold=threshold_hr, hyperparameter=hyperparameters[i])
        try:
            model.print_model_test_report()
        except:
            print("exception")


def recreate_reduced_all(grid_search=False, thresholds=[10, 15, 20]):
    feature_selection = [
        'mean',
        'number_zero_crossings',
        'kurtosis',
        'skewness',
        'hf_diff_acf',
        'hf_diff_data',
        'interval_lengths_std',
        'sqi_std',
        'sqi_min',
        'sqi_median',
        'peak_mean',
        'peak_std',
        'template_corr_highest_sqi_mean',
        'template_corr_highest_sqi_std',
        'template_corr_median_sqi_mean',
        'template_corr_median_sqi_std',
        'interval_means_std',
        'sqi_coverage_03',
        'sqi_coverage_04',
        'sqi_coverage_05'
    ]
    for threshold in thresholds:
        logging.info(f"Default segments, all features, threshold={threshold}")
        paths = get_paths(reduced=False, threshold=threshold)
        recreate_own_models(paths=paths, grid_search=grid_search)

        logging.info(f"Default segments, reduced feature set, threshold={threshold}")
        paths = get_paths(reduced=True, threshold=threshold)
        recreate_own_models(paths=paths, grid_search=grid_search, feature_selection=feature_selection)


def get_paths(reduced=False, segment_length=10, threshold=10):
    paths = ["RF_Clf", "RF_Regr", "XGB_Regr", "XGB_Clf"]
    paths = [path + "_s" + str(segment_length) for path in paths]
    if reduced:
        paths = [path + "_reduced_h" + str(threshold) for path in paths]
    else:
        paths = [path + "_all_" + str(threshold) for path in paths]
    return paths


if __name__ == "__main__":
    paths = ["RF_Clf_all_10", "RF_Regr_all_10", "XGB_Regr_all_10", "XGB_Clf_all_10"]
    recreate_own_models(paths=paths, grid_search=True)
    pass
