from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='funfolding',
    version='0.2.0',

    description='Having fun with unfolding.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/tudo-astroparticlephysics/funfolding',

    author='Mathis Boerner',
    author_email='mathis.boerner@tu-dortmund.de',

    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='unfolding',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'corner',
        'emcee==3.0rc1',
        'matplotlib',
        'numpy',
        'pymc3',
        'scikit-learn>=0.18.1',
        'scipy',
        'six>=1.1',
    ],
    extras_require={':python_version == "2.7"': ['futures']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
