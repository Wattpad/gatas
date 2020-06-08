from setuptools import setup, find_packages


setup(
    name='gatas',
    version='1.0.0',
    packages=find_packages(exclude=('tests', 'tests.*')),
    python_requires='~=3.7',
    install_requires=[
        'mlflow~=1.8',
        'defopt~=6.0',
        'numba~=0.49',
        'numpy~=1.18.0',
        's3fs~=0.4.0',
        'scikit-learn~=0.22.0',
        'tensorflow~=1.15',
    ])
