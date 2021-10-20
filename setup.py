from setuptools import setup
import os

libFolder = os.path.dirname(os.path.realpath(__file__))
requirementsPath = libFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementsPath):
    with open(requirementsPath) as f:
        install_requires = f.read().splitlines()

setup(name='KMeans-Image-Segmentation',
      install_requires=install_requires,
      version='0.1',
      description='K-Means Image Segmentation',
      url='https://github.com/mitchellsayer/KMeans-Image-Segmentation',
      author='Mitchell Sayer, ADD YOUR NAMES',
      author_email='mitchell.sayer@sjsu.edu',
      license='MIT',
      packages=['kmeans-imseg'],
      zip_safe=False)