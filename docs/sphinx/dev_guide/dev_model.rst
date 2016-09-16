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

The RAJA code lives in a `GitHub repository <https://github.com/LLNL/RAJA>`_.
The RAJA GitHub repository is the central interaction hub for RAJA developers.

======================================================
Branch Development Model
======================================================

This section describes the 'Gitflow' workflow model used by the RAJA project.
See `Atlassian Gitflow Description <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ for a more detailed 
description of Gitflow than what is provided here.

Gitflow defines a specific branch development model centered around project 
releases. It is a simple workflow that makes clear which branches are used 
for which phases of development and those phases are represented in the 
structure of the repository. Similar to other branch development models, 
developers using Gitflow do development locally and push their work to a 
central repository. The two main repository branches are *master* and 
*develop*. They always exist. Other branches are temporary. The master branch 
records the official release history. Each merge into master is tagged with 
a version number; see :ref:`versioning-label`. The develop branch is used to
integrate new features, bug fixes, etc. before they are pushed into master. 
Gitflow revolves around the distinction between these two branches.

Each new feature, or other well-defined chunk of work, is 
developed on its own branch, with changes being pushed to the central 
repository regularly for backup. Feature branches are created off the
develop branch. When a feature is complete, a pull request is submitted
for review and comment by other team members. When a sufficient number of 
reviewers have approved the pull request, the feature branch merged into
develop. **Feature branches never interact directly with the master branch.**

When the team has decided that enough features, bug fixes, etc. have been 
merged into the develop (for example, all items identified for a release),
a *release* branch is created off of develop to finalize the release. Creating
a release branch starts the next release cycle on develop. At this point, 
new work can start on feature branches for the next release. No new features
should be added to the release branch. Only bug fixes, documentation 
generation, and other release-oriented changes should go into the release 
branch. When the release branch is ready, it gets merged into master and 
master is tagged with a version number. Finally, master is merged back into 
develop which may have changed since the release was initiated.

Sometimes, there is a need for a *hotfix* branch to resolve an issue in
a released version. This is the only time a branch should be created off
master. When the fix is complete, it should be reviewed using a pull 
request and merged into master and develop. At this point, master should
be tagged with a new version number. The dedicated line of development for
bug fixes allows the team to quickly address issues without disrupting
other parts of the workflow. 

.. figure:: gitflow-workflow.png

   This example shows typical interactions between branches in the Gitflow 
   workflow. Here, master was merged into develop after tagging version v0.1. 
   A fix was needed and so a hotfix branch was created. When the fix was 
   completed, it was merged into master and develop. Also, master was tagged 
   with version v0.2. Work was performed on two feature branches. 
   When one feature was done, it was merged into develop. Then, a release 
   branch was created and it was merged into master when the release was
   finalized. Finally, master was tagged with version v1.0.

Summary of Gitflow workflow:

  * Features are developed, bugs are fixed, etc. on *feature* branches created
    off of the *develop* branch. When work is complete on a feature branch, 
    it is merged into develop.
  * At a release point, a *release* branch is created off of develop. When
    this is done, development can continue on develop for the next release.
    No features are added to a release branch -- only bug fixes, documentation,
    and other release-oriented changes go into a release branch. When the 
    release is ready, the release branch is merged into master and master is 
    tagged with a new version number. Master is also merged into develop at 
    this time.
  * Sometimes an issue needs to be addressed on master. This is done by 
    creating a *hotfix* branch off of master. When the fix is complete, the
    hotfix branch is merged into master and master is tagged with a new
    version number.


======================================================
Code Reviews and Acceptance
======================================================

Insert code review policy and pull request approval criteria here....

...reference test description here :ref:`testing-label`.


.. _versioning-label:

======================================================
Versioning
======================================================

This section describes the *semantic* versioning scheme used by the RAJA 
project. See `Semantic Versioning <semver.org>`_ for a more detailed 
description of semantic versioning than what is provided here.

Semantic versioning is a methodology for assigning version numbers to 
software releases in a way that conveys specific meaning about the code and
modifications made from version to version. Semantic versioning uses a
three part version number `xx.yy.zz`:

  * `xx` is the *major* version number. It changes when an incompatible API
    change is made. That is, the API changes in a way that may break code
    using an earlier release of the software with a smaller major version 
    number.
  * `yy` is the *minor* version number. It changes when functionality is
    added that is backward-compatible. The API may grow to access new 
    functionality. However, the software will function the same as any
    earlier release of the software with a smaller minor version number.
  * `zz` is the *patch* version number. It changes when a bug fix is made that
    is backward compatible. That is, such a bug fix is an internal 
    implementation change that fixes incorrect behavior.

Note that a key aspect of meaning for these three version numbers is that
the software has a public API. Changes to the API or code functionality
are communicated by the way the version number is incremented. Some important
conventions in the application of semantic versioning are:

  * Once a version of the software is released, the contents of the release 
    *must not* change. If the software is modified, is *must* be released
    as as a new version.
  * A major version of zero (i.e., `0.yy.zz`) is considered initial 
    development and indicates anything may change. The API is not considered
    stable.
  * Version `1.0.0` defines the public API. Version number increments beyond 
    this point depend on how the public API changes.
  * When the software is changed so that any API functionality becomes 
    deprecated, the minor version number *must* be incremented.
  * A pre-release version may be denoted by appending a hypen and a series
    of dot-separated identifiers after the patch version. For example,
    `1.0.1-alpha`, `1.0.1-alpha.1`, `1.0.2-0.2.5`.
  * Versions are compared using precedence that is calculated by separating
    major, minor, patch, and pre-release identifiers in that order. Major, 
    minor, and patch numbers are compared numerically from left to right. For 
    example, 1.0.0 < 2.0.0 < 2.1.0 < 2.1.1. When major, minor, and patch
    numbers are equal, a pre-release version has lower precendence. For 
    example, 1.0.0-alpha < 1.0.0.

By following these conventions, it is fairly easy to communicaet intent of
version changes to users and it will be (hopefully) straightforward for them
to manage dependencies on RAJA.
