.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see the RAJA/LICENSE file.
.. ##

*********************************
RAJA Development
*********************************

The RAJA code lives in a `GitHub repository <https://github.com/LLNL/RAJA>`_
and all development is done there.

======================================================
Branch Development Model
======================================================

The RAJA project follows the 'Gitflow' Workflow model for branch development.
See `Atlassian Gitflow Description <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ for a more detailed description that what 
is provided here.

The RAJA GitHub repository is the central interaction hub for RAJA developers. 
Developers work locally and push their work to the central repository. The two
main repository branches are 'master' and 'develop'. The master branch is used
to recod the official release history. Each commit or branch merge into master
is tagged with a version number; see :ref:`versioning-label`. The develop 
branch is the integration branch for new features, bug fixes, etc. before
they are pushed into master.


======================================================
Code Reviews and Acceptance
======================================================


.. _versioning-label:

======================================================
Versioning
======================================================
