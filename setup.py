from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


extras_require = {
    'tests': ['pytest'],
    'docs': ['sphinx', 'sphinx-rtd-theme'],
}
extras_require['all'] = extras_require['tests'] + extras_require['docs']

setup(
    name='funfolding',
    version='0.3.0',

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

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires=">= 3.9, <3.12a0",
    keywords='unfolding',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'corner',
        'emcee>=3.0',
        'matplotlib',
        'numpy',
        'pymc3',
        'scikit-learn>=0.19.0',
        'scipy',
    ],
    extras_require=extras_require,
)
