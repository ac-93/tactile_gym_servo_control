import os
from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))

# get required packages from requirements.txt
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(name='tactile_gym_servo_control',
      version='0.0.1',
      description='Conversion between tactile image gathered on a physical robot and those gathered in simulation',
      author='Alex Church, Yijiong Lin',
      author_email='alexchurch1993@gmail.com, yijionglin.bourne@gmail.com',
      license='MIT',
      packages=['tactile_gym_servo_control'],
      install_requires=install_requires,
      zip_safe=False)
