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
Gitflow Branching Model
======================================================

The RAJA team follows the 'Gitflow' branch development model, which is
summarized here. See the `Atlassian Gitflow Description <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ 
for more details.

Gitflow is a branching model centered around software 
releases. It is a simple workflow that makes clear which branches correspond
to which phases of development. In particular, those phases are represented 
explicitly in the structure of the repository. As in other branching models, 
developers develop code locally and push their work to a central repository. 
The two main repository branches are *master* and *develop*, which always 
exist. Other branches are temporary. The master branch records the official 
release history of the project. Each time the master branch is changed, it 
is tagged with a new version number. For a description of the RAJA versioning 
scheme, see :ref:`versioning-label`. The develop branch is used to
integrate new features and most bug fixes before they are merged into master. 
The distinction between these two main branches is central to Gitflow.

Each new feature, or other well-defined portion of work, is 
developed on its own branch, with changes being pushed to the central 
repository regularly for backup. Feature branches are created off the
develop branch. When a feature is complete, a pull request is submitted
for review by other team members. When all issues arising in a review 
have been addressed and reviewers have approved the pull request, the 
feature branch is merged into develop. **Feature 
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
tagged with a new version number. A dedicated line of development for
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

Before any code may be merged into the develop or master branches, it
must be tested, reviewed, and accepted. Submitting a pull request on
the RAJA Github project to merge a branch into develop or master initiates 
the test and review processes. 

Submitting a pull request, or modifying an existing one, triggers Travis and 
Appveyor CI RAJA builds and testing for various compilers and platforms 
including Windows. Similarly, builds and tests are invoked on various
production platforms at LLNL. All builds and tests must pass for a pull 
request to be approved. Also, it is expected that unit tests be constructed 
to exercise any new functionality that is introduced. This will be assessed
by reviewers of each pull request. See :ref:`testing-label` for more 
information about RAJA testing.

The code changes in a pull request must be accepted by at least two members
of the RAJA core development team. The changes are reviewed by the team
and are accepted, rejected, or commented on for improvement; e.g., 
issues to be addressed, suggested improvements, etc. Reviewers can accept a 
pull request using either the `pullapprove` Github plug-in or by giving a 
"thumbs up" in a Github pull request comment (type `:+1:` in the comment box).
When a pull request is approved, it can be merged. If the merged branch is
no longer needed for development, it should be deleted.

In addition to passing tests, changes to the develop and master branches
in RAJA should be scrutinized in other ways and using other tools. 
For example:

* The code should compile cleanly at the highest warning level with the 
  main compilers supported by the project. All warnings **must** be 
  understood and eliminated if possible. Reducing a compiler's warning 
  level to eliminate warning messages **is not** acceptable.

  Compiler warnings, while seemingly innocuous at times, often indicate
  problems that do not appear until later or until specific run-time
  conditions are encountered.

* Static analysis tools **should** be applied to the code using tools such
  as `cppcheck`, etc. to  

* Runtime memory checking (e.g., using Valgrind, cuda-memcheck, etc.) 
  **should** be used to verify that there are no leaks or other memory issues. 

We have not yet established policies or specific tools for code health 
analyses like these. Ideally, we would like to automate them as part of our
CI and pull request approval processes.  Please check back here in the future
for progress on this front.

---------------------------
Code Review Checklist
---------------------------

The following list contains important issues we want to catch and resolve 
when we review code in pull requests. They are not in any particular order. 
Most of these items appear in the coding guidelines; they are included here
for viewing as an easy checklist. The list will be modified based on what
the core developers believe are the main issues to look for when reviewing 
code.

 #. Blah...
 #. Blah-blah...
 #. Blah-blah-blah...


.. _versioning-label:

======================================================
RAJA Versioning
======================================================

The RAJA team follows the *semantic* versioning scheme, which is summarized
here. See `Semantic Versioning <semver.org>`_ for a more detailed description.

Semantic versioning is a methodology for assigning version numbers to 
software releases in a way that conveys specific meaning about the code and
modifications from version to version. Semantic versioning is based on a
three part version number `MM.mm.pp`:

  * `MM` is the *major* version number. It is incremented when an incompatible 
    API change is made. That is, the API changes in a way that may break code
    using an earlier release of the software with a smaller major version 
    number. Following Gitflow (above), the major version number may be changed
    when the develop branch is merged into the master branch.
  * `mm` is the *minor* version number. It changes when functionality is
    added that is backward-compatible. The API may grow to support new 
    functionality. However, the software will function the same as any
    earlier release of the software with a smaller minor version number
    when used through the intersection of two APIs. Following Gitflow (above), 
    the minor version number is always changed when the develop branch is 
    merged into the master branch, except possibly when the major version 
    is changed.
  * `pp` is the *patch* version number. It changes when a bug fix is made that
    is backward compatible. That is, such a bug fix is an internal 
    implementation change that fixes incorrect behavior. Following Gitflow 
    (above), the patch version number is always changed when a hotfix branch
    is merged into master, or when develop is merged into master and the 
    changes only contain bug fixes.

A key consideration in meaning for these three version numbers is that
the software has a public API. Changes to the API or code functionality
are communicated by the way the version number is incremented. Some important
conventions followed when using semantic versioning are:

  * Once a version of the software is released, the contents of the release 
    *must not* change. If the software is modified, it *must* be released
    as as a new version.
  * A major version number of zero (i.e., `0.mm.pp`) is considered initial 
    development where anything may change. The API is not considered stable.
  * Version `1.0.0` defines the first stable public API. Version number 
    increments beyond this point depend on how the public API changes.
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
