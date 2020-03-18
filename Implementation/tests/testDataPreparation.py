import unittest
import numpy as np
from data_preparation import BcgData, DataSeries


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = BcgData()

    def test_dataSeries_types(self):
        series = self.data.data_series[0]
        self.assertEqual(type(series), DataSeries)

        self.assertEqual(type(series.raw_data), np.ndarray)
        self.assertEqual(series.raw_data.ndim, 1)

        self.assertEqual(type(series.sqi), np.ndarray)
        self.assertEqual(series.sqi.ndim, 1)

        self.assertEqual(type(series.bbi_bcg), np.ndarray)
        self.assertEqual(series.bbi_bcg.ndim, 1)

        self.assertEqual(type(series.bbi_ecg), np.ndarray)
        self.assertEqual(series.bbi_ecg.ndim, 1)

        self.assertEqual(type(series.indices), np.ndarray)
        self.assertEqual(series.indices.ndim, 1)

        self.assertEqual(type(series.samplerate), int)


if __name__ == '__main__':
    unittest.main()
