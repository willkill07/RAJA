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
This GitHub repo is the central interaction hub for RAJA developers.

======================================================
Branching Model
======================================================

The RAJA team follows the 'Gitflow' branch development model, which is
summarized in this section. See the `Atlassian Gitflow Description <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ 
for a more detailed description than what is provided here.

Gitflow defines a specific branching model centered around software 
releases. It is a simple workflow that makes clear which branches correspond
to which phases of development. Those phases are represented explicitly in the 
structure of the repository. AS in other branching models, developers using 
Gitflow do development locally and push their work to a central repository. 
The two main repository branches are *master* and *develop*, which always 
exist. Other branches are temporary. The master branch records the official 
release history. Each time the master branch is changed, it is tagged with 
a new version number. For a description of the RAJA versioning scheme,
see :ref:`versioning-label`. The develop branch is used to
integrate new features and most bug fixes before they are merged into master. 
The distinction between these two main branches is central to Gitflow.

Each new feature, or other well-defined chunk of work, is 
developed on its own branch, with changes being pushed to the central 
repository regularly for backup. Feature branches are created off the
develop branch. When a feature is complete, a pull request is submitted
for review and comment by other team members. When reviewers have approved 
the pull request, the feature branch is merged into develop. **Feature 
branches never interact directly with the master branch.**

When the team has decided that enough features, bug fixes, etc. have been 
merged into develop (for example, all items identified for a release have
been completed), a *release* branch is created off of develop to finalize 
the release. Creating a release branch starts the next release cycle on 
develop. At that point, new work can start on feature branches for the 
next release. No new features are added to a release branch. Only bug fixes, 
documentation, and other release-oriented changes go into a release 
branch. When a release branch is ready, it is merged into master and 
master is tagged with a new version number. Finally, master is merged back 
into develop since it may have changed since the release was initiated.

Sometimes, there is a need for a *hotfix* branch to resolve an issue in
a released version. This is the only time a branch is created off of
master. When the fix is complete, it is reviewed using a pull request and 
then merged into both master and develop. At this point, master is
tagged with a new version number. The dedicated line of development for
bug fixes, using a hotfix branch, allows the team to quickly address issues 
without disrupting other parts of the workflow. 

.. figure:: gitflow-workflow.png

   This figure shows typical interactions between branches in the Gitflow 
   workflow. Here, master was merged into develop after tagging version v0.1. 
   A fix was needed and so a hotfix branch was created. When the fix was 
   completed, it was merged into master and develop. Master was tagged 
   with version v0.2. Also, work was performed on two feature branches. 
   When one feature branch was done, it was merged into develop. Then, a 
   release branch was created and it was merged into master when the release 
   was finalized. Finally, master was tagged with version v1.0.

Summary of Gitflow workflow:

  * Features are developed and most bugs are fixed on *feature* branches 
    created off of the *develop* branch. When work is complete on a feature 
    branch, it is merged into develop.
  * At a release point, a *release* branch is created off of develop. At this
    point, development can continue on develop for the next release.
    No features are added to a release branch -- only bug fixes, documentation,
    and other release-oriented changes go into a release branch. When a
    release is ready, the release branch is merged into master and master is 
    tagged with a new version number. Master is also merged into develop at 
    this time.
  * Sometimes an issue needs to be addressed on master. This is done by 
    creating a *hotfix* branch off of master. When the fix is complete, the
    hotfix branch is merged into master and develop and master is tagged 
    with a new version number.


======================================================
Code Reviews and Acceptance
======================================================

Insert code review policy and pull request approval criteria here....

...reference test description here :ref:`testing-label`.


.. _versioning-label:

======================================================
Versioning
======================================================

The RAJA team follows the *semantic* versioning scheme, which is summarized
in this section. See `Semantic Versioning <semver.org>`_ for a more detailed 
description.

Semantic versioning is a methodology for assigning version numbers to 
software releases in a way that conveys specific meaning about the code and
modifications made from version to version. Semantic versioning is based on a
three part version number `xx.yy.zz`:

  * `xx` is the *major* version number. It changes when an incompatible API
    change is made. That is, the API changes in a way that may break code
    using an earlier release of the software with a smaller major version 
    number.
  * `yy` is the *minor* version number. It changes when functionality is
    added that is backward-compatible. The API may grow to support new 
    functionality. However, the software will function the same as any
    earlier release of the software with a smaller minor version number
    when used through the intersection of two APIs.
  * `zz` is the *patch* version number. It changes when a bug fix is made that
    is backward compatible. That is, such a bug fix is an internal 
    implementation change that fixes incorrect behavior.

A key consideration of meaning for these three version numbers is that
the software has a public API. Changes to the API or code functionality
are communicated by the way the version number is incremented. Some important
conventions followed when using semantic versioning are:

  * Once a version of the software is released, the contents of the release 
    *must not* change. If the software is modified, it *must* be released
    as as a new version.
  * A major version number of zero (i.e., `0.yy.zz`) is considered initial 
    development and indicates anything may change. The API is not considered
    stable.
  * Version `1.0.0` defines the public API. Version number increments beyond 
    this point depend on how the public API changes.
  * When the software is changed so that any API functionality becomes 
    deprecated, the minor version number *must* be incremented.
  * A pre-release version may be denoted by appending a hyphen and a series
    of dot-separated identifiers after the patch version. For example,
    `1.0.1-alpha`, `1.0.1-alpha.1`, `1.0.2-0.2.5`.
  * Versions are compared using precedence that is calculated by separating
    major, minor, patch, and pre-release identifiers in that order. Major, 
    minor, and patch numbers are compared numerically from left to right. For 
    example, 1.0.0 < 2.0.0 < 2.1.0 < 2.1.1. When major, minor, and patch
    numbers are equal, a pre-release version has lower precedence. For 
    example, 1.0.0-alpha < 1.0.0.

By following these conventions, it is fairly easy to communicate intent of
version changes to users and it should be straightforward for users
to manage dependencies on RAJA.
