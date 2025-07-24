# gesture_control_app/setup.py
from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the list of requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gesture_control_app',
    version='0.1.0', # Your current version
    author='Parth Malik', # Replace with your name
    author_email='parth.email@example.com', # Replace with your email
    description='A console-based application for controlling your computer using hand gestures.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/gesture-control-app', # Replace with your GitHub repo URL
    packages=find_packages(where='src'), # Look for packages in the 'src' directory
    package_dir={'': 'src'}, # Tell setuptools that packages are under 'src'
    include_package_data=True, # Include non-code files specified in MANIFEST.in or setup.cfg
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'gesture_control=main:main', # This creates an executable 'gesture_control' that runs src/main.py's main()
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: System :: Hardware',
        'Topic :: Utilities',
    ],
    python_requires='>=3.8',
    # Data files (these are usually placed outside source code but can be included if part of package)
    # data_files=[
    #     ('config', ['config/app_config.json']),
    # ],
)