import unittest
import numpy as np
from data_preparation import BcgData, DataSeries


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = BcgData()
        self.series = self.data.data_series[0]

    def test__type_series(self):
        for series in self.data.data_series:
            self.assertEqual(type(series), DataSeries)

    def test__type_raw_data(self):
        self.assertEqual(type(self.series.raw_data), np.ndarray)
        self.assertEqual(self.series.raw_data.ndim, 1)
        self.assertGreater(len(self.series.raw_data), 1)

    def test__type_sqi(self):
        self.assertEqual(type(self.series.sqi), np.ndarray)
        self.assertEqual(self.series.sqi.ndim, 1)
        self.assertGreater(len(self.series.sqi), 1)

    def test__type_bbi_bcg(self):
        self.assertEqual(type(self.series.bbi_bcg), np.ndarray)
        self.assertEqual(self.series.bbi_bcg.ndim, 1)
        self.assertGreater(len(self.series.bbi_bcg), 1)

    def test__type_bbi_ecg(self):
        self.assertEqual(type(self.series.bbi_ecg), np.ndarray)
        self.assertEqual(self.series.bbi_ecg.ndim, 1)
        self.assertGreater(len(self.series.bbi_ecg), 1)

    def test__type_indices(self):
        self.assertEqual(type(self.series.indices), np.ndarray)
        self.assertEqual(self.series.indices.ndim, 1)
        self.assertGreater(len(self.series.indices), 1)

    def test__type_samplerate(self):
        self.assertEqual(type(self.series.samplerate), int)

    def test__equal_array_length(self):
        self.assertEqual(len(self.series.sqi), len(self.series.bbi_ecg))
        self.assertEqual(len(self.series.sqi), len(self.series.bbi_bcg))
        self.assertEqual(len(self.series.sqi), len(self.series.indices))


if __name__ == '__main__':
    unittest.main()
