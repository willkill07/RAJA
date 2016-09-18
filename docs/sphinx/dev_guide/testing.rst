.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see the RAJA/LICENSE file.
.. ##

.. _testing-label:

*********************************
RAJA Testing
*********************************

RAJA developers use three different test suites to verify RAJA is correct
and performs as expected.

======================================================
Unit Tests
======================================================

The RAJA repository contains a collection of simple unit tests that
exercise basic functionality for all constructs and execution policies
that RAJA supports. These test provide a `zeroth-order` check to make sure 
RAJA is built correctly and nothing obvious is broken. These tests are 
built by default when RAJA is compiled. To execute the unit tests, simply 
enter the directory where RAJA is built and run them. For example:

.. code-block:: sh

    $ cd my-RAJA-build   // enter directory where RAJA is configured
    $ make               // build RAJA
    $ make test          // run tests

RAJA unit tests use CTest. So when the tests are run as
above, the output of individual tests is suppressed. Specifically, you will 
only see summary output indicating whether each unit test set passed or failed. 
More detailed information about individual tests can be observed by running
them directly and individually. The unit test executables are located 
in sub-directories of the test directory in the RAJA build space.

**It is important to note that the number of unit tests run varies
depending on how RAJA is configured for compilation. As more programming
model backend options are enabled, for example, more execution policies will 
be exercised; thus, more tests will be run.**


======================================================
Integration Tests
======================================================

More complex tests that begin to represent RAJA integration into an 
application are provided in the 
`RAJA example repository <https://github.com/LLNL/RAJA-examples>`_.
This repo contains RAJA variants of several proxy applications, such
as `LULESH <https://codesign.llnl.gov/lulesh.php>`_ and 
`Kripke <https://codesign.llnl.gov/kripke.php>`_. It also includes the 
original released versions of the proxies for code and performance comparison.

Directions for building and running the example tests are included in
the RAJA example repo.


======================================================
Performance Tests
======================================================

The RAJA performance suite is under construction....
