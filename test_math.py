import unittest


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

    def test_multiply(self):
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-2, 5), -10)
        self.assertEqual(multiply(0, 100), 0)


if __name__ == "__main__":
    unittest.main()
