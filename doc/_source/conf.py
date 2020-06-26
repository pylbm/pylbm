# -*- coding: utf-8 -*-
#
# pylbm documentation build configuration file, created by
# sphinx-quickstart on Wed Dec 11 10:32:28 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
sys.path.insert(0, os.path.abspath('../_sphinxext'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'numpydoc',
              'matplotlib.sphinxext.plot_directive',
              'lbm_ext',
              'nbsphinx',
              'jupyter_sphinx',
              'IPython.sphinxext.ipython_console_highlighting',
              'sphinx.ext.autosummary',
              ]
nbsphinx_execute = 'auto'              
#mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
#mathjax_path="/Users/graille/.ipython/nbextensions/mathjax/unpacked/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
#mathjax_path="/home/gouarin/.ipython/nbextensions/mathjax/unpacked/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
#autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'pylbm'
copyright = u'2020, Benjamin Graille, Loïc Gouarin'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import pylbm
version = pylbm.__version__
# The full version, including alpha/beta/rc tags.
release = pylbm.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['../_build', '**.ipynb_checkpoints']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'pylbm'
html_theme_path = ['../_theme']

html_title = "%s v%s Manual" % (project, version)
html_static_path = []
html_last_updated_fmt = '%b %d, %Y'

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

pngmath_use_preview = True
pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'agogo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "../_theme/pylbm/static/img/pylbm.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['../_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}
# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': ['navigation.html',
           'localtoc.html',
           #'versions.html'
          ],
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'pylbmdoc'

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
'preamble': '''
\\usepackage{amssymb}
\\newcommand{\\DdQq}[2]{{\\mathrm D}_{#1}{\\mathrm Q}_{#2}}
\\newcommand{\\drondt}{\\partial_t}
\\newcommand{\\drondtt}{\\partial_{tt}}
\\newcommand{\\drondx}{\\partial_x}
\\newcommand{\\drondxx}{\\partial_{xx}}
\\newcommand{\\drondy}{\\partial_y}
\\newcommand{\\drondyy}{\\partial_{yy}}
\\newcommand{\\dx}{\\Delta x}
\\newcommand{\\dt}{\\Delta t}
\\newcommand{\\grandO}{{\\mathcal O}}
\\newcommand{\\density}[2]{\\,f_{#1}^{#2}}
\\newcommand{\\fk}[1]{\\density{#1}{\\vphantom{\\star}}}
\\newcommand{\\fks}[1]{\\density{#1}{\\star}}
\\newcommand{\\moment}[2]{\\,m_{#1}^{#2}}
\\newcommand{\\mk}[1]{\\moment{#1}{\\vphantom{\\star}}}
\\newcommand{\\mke}[1]{\\moment{#1}{e}}
\\newcommand{\\mks}[1]{\\moment{#1}{\\star}}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'pylbm.tex', u'pylbm Documentation',
   u'Benjamin Graille, Loïc Gouarin', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'pylbm', u'pylbm Documentation',
     [u'Benjamin Graille, Loïc Gouarin'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'pylbm', u'pylbm Documentation',
   u'Benjamin Graille, Loïc Gouarin', 'pylbm', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'


# -- Options for Epub output ---------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = u'pylbm'
epub_author = u'Benjamin Graille, Loïc Gouarin'
epub_publisher = u'Benjamin Graille, Loïc Gouarin'
epub_copyright = u'2020, Benjamin Graille, Loïc Gouarin'

# The language of the text. It defaults to the language option
# or en if the language is not set.
#epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
#epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#epub_identifier = ''

# A unique identification for the text.
#epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
#epub_cover = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_post_files = []

# A list of files that should not be packed into the epub file.
#epub_exclude_files = []

# The depth of the table of contents in toc.ncx.
#epub_tocdepth = 3

# Allow duplicate toc entries.
#epub_tocdup = True

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
numpydoc_use_plots = True
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = False
plot_formats = [('png', 100), 'pdf']

import math
phi = (math.sqrt(5) + 1)/2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}

autosummary_generate = True