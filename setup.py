from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='rudinn@ethz.ch',
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'numpy==1.23.5',
                      'pandas==2.0.3',
                      'setuptools==59.5.0',
                      'tensorboard==2.14.0',
                      'wandb',
                      'pybullet',
                      'viser']
                      
)