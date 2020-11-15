from data_statistical_features import *
from own_models_train import *


def reproduce_own_segments(segment_lengths=[5, 10, 20, 30], overlap_amounts=[0.8, 0.9, 0.9, 0.9],
                           data_folder='data_patients'):
    assert len(segment_lengths)==len(overlap_amounts)
    for segment_length, overlap_amount in zip(segment_lengths, overlap_amounts):
        DataSetOwn(segment_length=segment_length, overlap_amount=overlap_amount, data_folder=data_folder)


if __name__ == "__main__":
    reproduce_own_segments(segment_lengths=[10], overlap_amounts=[0.9], data_folder='data_healthy')
    reproduce_own_segments(segment_lengths=[5, 10, 20, 30], overlap_amounts=[0.8, 0.9, 0.9, 0.9],
                           data_folder='data_patients')
    get_final_models(grid_search=False, segment_length=10, overlap_amount=0.9, threshold_hr=10)
    get_final_models(grid_search=False, segment_length=10, overlap_amount=0.9, threshold_hr=5)
    get_final_models(grid_search=False, segment_length=10, overlap_amount=0.9, threshold_hr=20)
    get_final_models(grid_search=False, segment_length=10, overlap_amount=0.9, threshold_hr=15)
    get_final_models(grid_search=False, segment_length=5, overlap_amount=0.8, threshold_hr=20)
    get_final_models(grid_search=False, segment_length=20, overlap_amount=0.9, threshold_hr=20)
    get_final_models(grid_search=False, segment_length=30, overlap_amount=0.9, threshold_hr=20)
