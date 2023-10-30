import unittest

def run_tests(test_directory="tests"):
  print("-" * 70)
  # Use the default test discovery provided by unittest
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover(test_directory)

  # Run the tests
  test_runner = unittest.TextTestRunner(verbosity=2)
  result = test_runner.run(test_suite)

  # Check the test result
  print("-" * 70)
  print(f"Total tests run: {result.testsRun}")
  if result.wasSuccessful():
    print("All tests passed!")
  else:
    print("Some tests failed.")
    print(f"Tests passed: {result.testsRun - len(result.failures)}")
    print(f"Tests failed: {len(result.failures)}")
    print("Failed tests:")
    for failure in result.failures:
      test_case, traceback = failure
      print(f"- Test case: {test_case.id()}")
      print(f"  Error message: {traceback}")
  print("-" * 70)

if __name__ == "__main__":
  run_tests()
