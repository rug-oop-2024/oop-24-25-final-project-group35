import unittest
# flake8: noqa: E402
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline
from autoop.tests.test_metrics import TestMetrics
from autoop.tests.test_classification import TestClassification

if __name__ == '__main__':
    unittest.main()
