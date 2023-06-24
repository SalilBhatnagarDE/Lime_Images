# from distutils.core import setup
import setuptools

# from setuptools import find_packages
setuptools.setup(
    name="Lime_Images",
    include_package_data=True,
    version='0.1',
    author="Salil Bhatnagar",
    author_email="salil.bhatnagar@fau.de",
    packages=setuptools.find_packages(),
    install_requires=["torch", "torchvision", "matplotlib", "numpy",
                      "urllib3", "opencv-python", "python-math", "Pillow", "scikit-image", "scikit-learn"]
)
