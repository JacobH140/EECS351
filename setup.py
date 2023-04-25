from setuptools import find_packages, setup

setup(
    name='EarReaderSSD',
    version='1.0',
    packages=find_packages(),
    url='',
    license='',
    author='jacobhume',
    author_email='jakehume@umich.edu',
    py_modules=['playback', 'sonify_character', 'sonify_sentence', 'sonify_text', 'sonify_word', 'syntax_tree_v2'],
    install_requires=['numpy', 'pyfiglet', 'pandas', 'nltk', 'mido', 'music21', 'pyo'],
    description=''
)
