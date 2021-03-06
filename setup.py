#!/usr/bin/python3

# from Cython.Build import cythonize
from setuptools import setup
# import numpy
import os

# Read the long description from the readme file
with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")


# Read the dependencies
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = os.path.join(thelibFolder, 'requirements.txt')
install_requires_list = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires_list = f.read().splitlines()


# Run setup
setup(name='ARLreader',
      packages=['ARLreader', 'tests'],
      package_data={},

      version='0.1.0',
      description='Package for downloading and extracting meteorological' +
                  ' profiles from GDAS1 global dataset through command line.',
      long_description=long_description,
      url='https://github.com/martin-rdz/ARLreader',
      download_url='https://github.com/martin-rdz/ARLreader/archive/master.zip',
      author='Martin Radenz',
      license='MIT',
      include_package_data=False,
    #   include_dirs=[numpy.get_include()],
    #   ext_modules = cythonize("ARLreader/fast_funcs.pyx"),
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Intended Audience :: Science/Research',
      ],
      keywords='GDAS1 ARL Hysplit',
      install_requires=install_requires_list,
      entry_points={
          'console_scripts': [
              'ARLreader=ARLreader:main',
          ],
      },
      )
