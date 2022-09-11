# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import sys
import os
#sys.path.append(os.getcwd().strip("/docs/source"))
#sys.path.insert(1,os.path.abspath("..."))
sys.path.append("/home/ilmari/python/moska")
print(sys.path)
#import Moska
#import Deck,Game,Hand,Player,Turns,utils
#import Deck,Game,Hand,Player,Turns,utils
#from ...Moska import Deck,Game,Hand,Player,Turns,utils


project = 'Moska'
copyright = '2022, Ilmari Vahteristo'
author = 'Ilmari Vahteristo'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
