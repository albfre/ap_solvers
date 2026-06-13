from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = [
  'mpmath>=1.2',
]
test_requirements = [
  'pytest>=7.0',
  'scipy',
  'numpy',
  'parameterized',
]

setup(
  author="Albin Fredriksson",
  author_email='albin.fredriksson@gmail.com',
  python_requires='>=3.9',
  install_requires=requirements,
  license="BSD-3-Clause",
  long_description=readme,
  long_description_content_type='text/markdown',
  include_package_data=True,
  name='ap_solvers',
  packages=find_packages(include=['ap_solvers', 'ap_solvers.*']),
  test_suite='tests',
  tests_require=test_requirements,
  url='https://github.com/albfre/ap_solvers',
  version='0.1.0'
)
