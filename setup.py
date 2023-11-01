from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = []
test_requirements = []

setup(
  author="Albin Fredriksson",
  author_email='albin.fredriksson@gmail.com',
  python_requires='>=3.6',
  install_requires=requirements,
  license="BSD license",
  long_description=readme,
  include_package_data=True,
  name='ap_solvers',
  packages=find_packages(include=['ap_solvers', 'ap_solvers.*']),
  test_suite='tests',
  tests_require=test_requirements,
  url='https://github.com/albfre/mpmath-solvers',
  version='0.1.0'
)
