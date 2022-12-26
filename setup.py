from setuptools import setup, find_packages

setup(
    name='mzk-karticky',
    packages=[
        'src',
        'src/alignment',
        'src/label_studio',
        'src/NER',
        'src/lambert',
        'src/matching',
        'src/old',
        'src/bert'
    ],
)
