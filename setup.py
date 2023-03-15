from setuptools import setup, find_packages

setup(
    name='flow_project',
    version='0.0.1',
    author='',
    author_email='',
    description='Optical Flow Estimation Project',
    url='https://github.com/GGomez99/Optical-Flow-Motion-Estimation-Project',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch', 'torchvision'],
)