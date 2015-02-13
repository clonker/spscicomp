.. spscicomp documentation master file, created by
   sphinx-quickstart2 on Thu Jan 15 18:29:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to spscicomp's documentation!
=====================================

Contents:

.. toctree::
   :maxdepth: 2

   common
   kmeans
   hmm
   tica


==============================
How to build the documentation
==============================

1. Merge your code branch into the dev branch. ::

    $ git checkout dev
    $ git merge <YOUR BRANCH>
  
2. Edit the rst files in docs/ to include your documentation in the code. Python, NumPy and Google style documentation markup is supported.
3. In the docs/ folder, run the compilation: ::

    $ make html
  
4. Commit the documentation changes. Be careful to not include any other changes you may have done, as we will cherry-pick this commit into the GitHub Pages branch. ::

    $ git add docs/
    $ git commit -m "updated documentation."
  
5. Switch to the gh-pages branch and cherry-pick the commit into this branch. ::

    $ git checkout gh-pages
    $ git cherry-pick <COMMIT ID>
  
6. Push! ::

    $ git push


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

