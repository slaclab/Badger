from setuptools import setup, find_packages

setup(
    name='badger-opt',
    version='0.1',
    description='Core of the Badger optimizer',
    url='https://github.com/SLAC-ML/Badger',
    author='Zhe Zhang',
    author_email='zhezhang@slac.stanford.edu',
    license='GPL',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pyyaml',
        'coolname',
        'PyQt6'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'badger = badger.__main__:main'
        ]
    },
)
