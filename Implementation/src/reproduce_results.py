from data_statistical_features import *


def reproduce_own_segments(segment_lengths=[5, 10, 20, 30], overlap_amounts=[0.8, 0.9, 0.9, 0.9]):
    assert len(segment_lengths)==len(overlap_amounts)
    for segment_length, overlap_amount in zip(segment_lengths, overlap_amounts):
        DataSetOwn(segment_length=segment_length, overlap_amount=overlap_amount)


if __name__ == "__main__":
    reproduce_own_segments(segment_lengths=[5, 20, 30], overlap_amounts=[0.8, 0.9, 0.9])
