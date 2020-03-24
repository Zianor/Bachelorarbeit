import unittest
from ml_statistical_features import DataSet


class MyTestCase(unittest.TestCase):
    def test_seconds_to_frames(self):
        self.assertEqual(DataSet._seconds_to_frames(1.003, 100), int(1.003*100))


if __name__ == '__main__':
    unittest.main()
