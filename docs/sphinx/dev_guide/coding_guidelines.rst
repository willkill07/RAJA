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
RAJA Coding Guidelines
*********************************

======================================================
0 Purpose of the guidelines and general considerations
======================================================

These guidelines define code style conventions for RAJA. Most of the 
guidelines were taken from the cited references, sometimes with 
modifications and simplifications; see :ref:`codingrefs-label`.

The guidelines emphasize code readability, correctness, portability, and 
interoperability. Agreement on coding style and following common idioms 
and patterns provides many benefits to a project with multiple developers. 
A uniform "look and feel" makes it easier to read and understand source code, 
which increases team productivity and reduces confusion and coding errors 
when developers work with code they did not write. Also, guidelines 
facilitate code reviews by enforcing consistency and focusing developers on 
common concerns. Some of these guidelines are arbitrary, but all are based 
on practical experience and widely accepted sound practices. For brevity, 
most guidelines contain little detailed explanation or justification. 

Each guideline is qualified by one of three auxiliary verbs: 
"must", "should", or "may". 

* A "must" item is an absolute requirement. 
* A "should" item is a strong recommendation. 
* A "may" item is a potentially beneficial stylistic suggestion. 

Whether and how to apply items qualified with "should" or "may" often depends
on the particular code situation. It is best to use them in a manner that
enhances code readability and help to reduce user and developer errors.


=========================================================
1 General guidelines
=========================================================

1.1 When modifying existing code, the style conventions already in
use in the file **must** be followed. This is not intended to
stifle personal creativity - mixing style is disruptive and 
may cause confusion for users and fellow developers.

1.2 Significant deviations from these guidelines **must** be discussed and
agreed upon by the development team. If the guidelines need to be changed,
they **should** be.


========
2 Names
========

This section contains guidelines for naming files, types. functions, 
class members, variables, etc. The main goal is to use a distinct and 
consistent naming convention for each item so that the role of each entity 
in the code is obvious from the form of its name.

-----------
2.1 General
-----------

2.1.1 Good names are essential to sound software design. Terms **must** be 
clear and well-defined, and terminology **must** be used consistently; 
i.e., in documentation and for names of directories, types, functions, 
variables, etc. Multiple terms **should** not be used to refer to the 
same concept and a concept **should** not be referred to by multiple 
terms.

      Using a clear, limited set of terminology in a software project helps
      maintain the consistency and integrity of the software, and it makes
      the code easier to understand for developers and users.

2.1.2 Every name **must** be meaningful. Its meaning **must** be clear
to other code developers and users, not just the author of the name.

      A substantial benefit of good name selection is that it can greatly
      reduce the amount of developer debate to define a concepts. A good name
      also tends to reduce the amount of documentation required for others to
      understand it. For example, when the name of a function clearly indicates
      what it does and the meaning and purpose of each argument is clear from
      its name, then code comments may be unnecessary. Documentation is
      can be a substantial part of software and requires maintenance. 
      Minimizing the amount of documentation required reduces this burden.

2.1.2 Each name **must** be consistent with other similar names in the code.

      For example, if getter/setter methods follow the convention "getFoo"
      and "setFoo" respectively, then adding a new setter method called
      "putBar" is clearly inconsistent.

2.1.3 Tersely abbreviated or cryptic names **should** be avoided. However, 
common acronyms and jargon that are well understood by team members and
users **may** be used.


--------------------
2.2 Directory names
--------------------

2.2.1 Each source directory **must** be named so that the collective purpose 
of the files it contains is clear. Directory names **should** follow
the same style conventions. 

      RAJA directory names use all lower case letters and consist of a single 
      word in most cases. A directory name with more than one word uses 
      a 'hyphen' to separate words.


--------------
2.3 File names
--------------

In this section and throughout the guidelines, we refer to "header" and
"source" files. Header files are those that are included in other files
(either header or source). Source files are not included in other files and
form distinct translation units when a program is compiled.

2.3.1 C++ header and source file extensions **must** be: \*.hxx and \*.cxx, 
respectively.

2.3.2 Header and source files that are closely related, such as a header file
containing prototypes for a set of methods and a source file containing
implementations of those methods **must** be named the same (excluding 
file extension, of course) or sufficiently similar so that their 
relationship is clear.

2.3.3 The name of each file **must** clearly indicate its contents.

      For example, the header and source file containing the definition and
      implementation of a major type, such as a class **must** include the 
      type name of the type in the file name. For example, the header and
      implementation file for a class called "MyClass" should be named 
      "MyClass.hxx" and "MyClass.cxx", respectively.

      Files that are not associated with a single type, but which contain 
      closely related functionality or concepts, **must** be named so that
      the functionality or concepts are clear from the name. For example,
      files that define and implement methods that handle file I/O **should** 
      be named "FileIO.hxx" and "FileUtils.cxx", or similar.

2.3.4 File names that differ only in letter case **must** not be used.

      Since we aim to support Windows platforms, which has limited case
      sensitivity for file names, having files with names "MyClass.hxx" 
      and "myclass.hxx", for example, is not acceptable. 


------------------------
2.4 Scope and type names
------------------------

2.4.1 Type names (i.e., classes, structs, typedefs, enums, etc.) **must** be 
nouns and **should** be in mixed case with each word starting with 
an upper case letter and all other letters in lower cases.

      For example, these are preferred type names::

         IndexSet, RangeSegment, PolicyBase

      These type names should not be used::

         indexSet, rangesegment, POLICYBASE

2.4.2 Separating characters, such as underscores, **should** not be used 
between words in a type name.

      For example, these names are not preferred type names::

         Index_set, Range_Segment

      **Exceptions to the guidelines above** include cases where types
      play a similar role to those in common use elsewhere. For example, RAJA 
      has iterator classes, such as "base_iterator" and "numeric_iterator". 
      These names are acceptable since they are consistent with those found 
      in the C++ standard library.

2.4.3 Suffixes that may be used by compilers for name mangling, or 
which are used in the C++ standard library, such as "\_t", **must** not 
be used in RAJA type names.


------------------------
2.5 Function names
------------------------

2.5.1 Each function **must** be named to clearly indicate what it does.

2.5.2 Function names **should** begin with a verb.

2.5.3 Complementary verbs such as  "get/set", "add/remove" and "create/destroy"
**must** be used for routines that perform complementary operations.

      Such symmetry prevents confusion and makes interfaces easier to use.

2.5.4 Verbs such as "is", "has", "can", etc. **should** be used for functions 
with a boolean return type.

      For example, the following names are preferred::

         isInitialized(), isAllocated()

2.5.5 Function names **must** use "camelCase" or "pot_hole" style.

      **camelCase style:** The first word has all lower case letters.
      If multiple words are used, each word after the first starts with
      an upper case letter and all other letters in the word are lower case.
      Underscores must not be used in camelCase names, but numbers may be used.

      For example, these are proper camelCase names::

         getLength(), createView2()

      **pot_hole style:** All letters are lower case. If multiple
      words are used, they are separated by a single underscore. Numbers
      may be used in pothole style names.

      For example, these are acceptable pothole style variable names::

         push_front(), push_back_2()

2.5.6 Names of related functions, such as methods for a class, **should** 
follow the same style.
 
      **Exception:** While consistency is important, name style may be mixed 
      when it makes sense to do so. For example, most methods for a class may 
      follow camelCase style. But, that same class may also contain methods 
      that follow pot_hole style if those methods perform operations that are
      similar to C++ standard library functions.


-----------------------------------
2.6 Data member and variable names
-----------------------------------

2.6.1 Variables that are function arguments or function-scoped variables 
**must** use either "camelCase" style or "pot_hole" style. Pot_hole style 
is preferred; camelCase is acceptable. 

       For example, these are acceptable variable names::

         myAverage, person_name, pressure2

2.6.2 Non-static class and struct data member names **must** have the 
prefix "m\_".

      This convention makes it obvious which variable names in the code refer
      to class members/struct fields and which are other local variables. For 
      example, the following are acceptable names for class data members using
      camelCase style::

         m_myAverage, m_personName

      and acceptable pothole style::

         m_my_average, m_person_name

2.6.3 Static class and struct data member names and static file scope variables
**must** have the prefixes "s\_".

      Similar to the guideline above, this makes it obvious that the variable
      is static.

2.6.4 Verbs, such as "is", "has", "can", etc., **shuld** be used for boolean 
variables (i.e., either type bool or integer that indicates true/false).

      For example, these names are preferred::

         m_is_initialized, has_license

      to these names::

         m_initialized, license

2.6.5 A variable that refers to a non-fundamental type **should** give an 
indication of its type.

      For example,::

         Topic* my_topic;

      is clearer than::

         Topic* my_value;


------------------------------------
2.7 Macros and enumeration constants
------------------------------------

2.7.1 Preprocessor macro constants **must** be named using all uppercase 
letters and underscores should be used between words.

      For example, these are acceptable macro names::

         MAX_ITERATIONS, READ_MODE

      These are not acceptable::

         maxiterations, readMode

2.7.2 The name of each enumeration value **should** start with a capital letter
and use an underscore between words when multiple words are used.

       For example,::

          enum Orange
          {
             Navel,
             Valencia,
             Num_Orange_Types
          };


=====================================
3 Directories, Files, and Scope
=====================================

This section contains basic directory and file organization guidelines.
These guidelines help make it easy to locate a file and
to locate essential information in a file easily and quickly.

----------------
3.1 Directories
----------------

3.1.1 The contents of each directory and file **must** be well-defined and
limited so that it can be named to clearly indicate its contents. 
The goal is to prevent directories and files from becoming bloated with 
too many or diverse concepts.


-------------------------------------
3.2 File location
-------------------------------------

3.2.1 Header files and associated implementation files **should** reside in 
the same directory, which is common practice for C++ libraries, unless
there is a good reason to do otherwise.

3.2.2 Each file **must** reside in the directory that corresponds to the code 
functionality supported by the contents of the file.


---------------------------------------------------------
3.3 General header file guidelines
---------------------------------------------------------

Consistently-applied conventions and header file organization can significantly
improve user understanding and developer productivity. This section provides 
general header file guidelines. In section :ref:`headerorg-label`, we describe 
recommended header file organization.

3.3.1 A header file **may** contain multiple type definitions (e.g., structs, 
classes, enums, etc.). However, type definitions and function declarations in 
a header file **must** be related closely and/or support the primary type for 
which the file is named.

3.3.2 A header file **must** be self-contained and self-sufficient.

      Specifically, a header file

      * **Must** have proper header file include guards 
        (see :ref:`headerorg-label`) to prevent multiple inclusion. The macro 
        symbol name for each guard must be chosen to guarantee uniqueness 
        within a compilation unit.
      * **Must** include all other headers and/or forward declarations it 
        needs to be compiled standalone. In addition, a file **should not** 
        rely on symbols defined nother header files it includes; the 
        other files **should** be included explicitly.
      * **Must** contain implementations of all generic templates and inline
        methods defined in it. A compiler will require the full definitions of
        these constructs to be seen in every source file that uses them.

        **Note:** Function templates or class template members whose 
        implementations are fully specialized with all template arguments 
        **must** be defined in an associated source file to avoid linker 
        errors. Fully specialized templates are not templates and 
        are treated just like regular functions.

3.3.3 Header files **should** use forward declarations instead of header file 
inclusions when possible. This prevents a compiler from opening more files 
than it needs to, which can speed up recompilation when header files change.

      **Exceptions:**

      * Header files that define external APIs for RAJA **must**
        include all header files for all types that appear in the API. This
        makes use of the API much easier.
      * When using a function, such as an inline method or template, that is
        implemented in a header file, the header file containing the
        implementation **must** be included.
      * Similarly, when using C++ standard library types in a header file, it
        **may** be preferable to include the actual headers (rather than 
        forward reference headers (e.g., 'iosfwd') in the header file to 
        make it easier to use. This avoids having explicit
        inclusion of standard headers wherever the header file is used.

3.3.4 A forward type declaration **must** be used in a header file when an 
include statement would result in a circular dependency among header files 
or when only the type name is needed and not the type definition.

3.3.5 Extraneous header files or forward declarations (i.e., those not 
required for standalone compilation) **should not** be included in header files.

      Spurious header file inclusions, in particular, introduce spurious file
      dependencies, which can increase compilation time unnecessarily.

3.3.6 Header file include statements **should** use the same ordering pattern 
for all files.

      This improves code readability, helps to avoid misunderstood
      dependencies, and insures successful compilation regardless of
      dependencies in other files. A common, recommended header file 
      inclusion ordering scheme is:

      1. Related header (e.g., class header in class implementation file)
      2. C library headers (if needed)
      3. C++ library headers
      4. Headers from other libraries
      5. Project headers

      Also, code is easier to understand when include files are ordered
      alphabetically within each of these sections and a blank line is
      inserted between sections. Also, adding comments that describe the
      header file categories are sometimes useful.  For example,

.. code-block:: cpp

         // Related header
         #include "MyClass.hpp"

         // C standard library (including non-std unistd.h)
         #include <stdio.h>
         #include <unistd.h>

         // C++ standard library
         #include <unordered_map>
         #include <vector>

         // "base" library headers
         #include "base/Port.hxx"

         // Headers from this project
         #include "MyOtherClass.hpp"

3.3.7 A "typedef" statement, when used, **should** appear in the header file 
where the type is defined. 

      This practice helps insure that all names associated with a given type
      are available when the appropriate header file is used and eliminates
      potentially inconsistent type names.

3.3.8 Routines **should** be ordered and grouped in a header file so that
code readability and understanding are enhanced.

      For example, all related methods should be grouped together.

3.3.9 The name of each function argument **must** be specified in a header 
file declaration. Also, names in function declarations and definitions 
**must** match.

       For example, this is not an acceptable function declaration::

          void doSomething(int, int, int);

3.3.10 Each function, type, and variable declaration in a header file **must** 
be documented according to the guidelines in Section 4.

       **Exception:** A set of short, simple functions (e.g., inline functions)
       with related functionality **may** be grouped together and described 
       with a single documetation prologue if the result is clearer and more 
       concise documentaion. Good names that are self-explanatory are 
       generally preferable to writing (and maintaining!) documentation.


.. _headerorg-label:

---------------------------------------------------------
3.4 Header file (\*.hxx extension) content organization
---------------------------------------------------------

Content **must** be organized consistently in all header files. The file 
layout described here is recommended. The following summary uses numbers 
and text to illustrate the basic structure. Details about individual items 
are contained in the guidelines after the summary.

.. code-block:: cpp

   // (1) Doxygen file prologue

   // (2a) Header file include guard, e.g.,
   #ifndef MYCLASS_HPP
   #define MYCLASS_HPP

   // (3) RAJA copyright and release statement

   // (4) Header file inclusions (when NEEDED in lieu of forward declarations)
   #include "..."

   // (5) Forward declarations NEEDED in header file (outside of RAJA namespace)
   class ...;

   // (6a) RAJA namespace declaration
   namespace RAJA {

   // (7a) RAJA internal namespace (if used); e.g.,
   namespace awesome {

   // (8) Forward declarations NEEDED in header file (in RAJA namespace(s))
   class ...;

   // (9) Type definitions (class, enum, etc.) with Doxygen comments e.g.,
   /*!
    * \brief Brief ...summary comment text...
    *
    * ...detailed comment text...
    */
   Class MyClass {
      ...
   };

   // (7b) RAJA internal namespace closing brace (if needed)
   } // awesome namespace closing brace

   // (6b) RAJA namespace closing brace
   } // RAJA namespace closing brace

   // (2b) Header file include guard closing endif */
   #endif // closing endif for header file include guard

The numbered item in the summary above are referred to in the following 
guidelines.

3.4.1 Each header file **must** begin with a Doxygen file prologue (item 1).

      See Section 4 for details.

3.4.2 The contents of each header file **must** be guarded using a 
preprocessor directive that defines a unique "guard name" for the header.

      The guard must appear immediately after the file prologue and use the
      '#ifndef' directive (item 2a); this requires a closing '#endif' 
      statement at the end of the file (item 2b). The preprocessor constant 
      must use the file name followed by "_HPP"; e.g., "MYCLASS_HPP" as above.

3.4.3 Each header file **must** contain a comment section that includes the 
RAJA copyright and release statement (item 3).

      See Section 4 for details.

3.4.4 All necessary header file inclusion statements (item 4) **must** 
appear immediately after copyright and release statement and before any 
forward declarations, type definitions, etc.

3.4.5 Any necessary forward declarations (item 5) for types defined outside 
the RAJA namespace **must** appear after the header include statements
and before the RAJA namespace statement.

3.4.6 All types defined and methods defined in a header file **must** be 
included in a namespace.

      Either the main "RAJA" namespace (item 6) or a namespace
      nested within the RAJA namespace (item 7) may be used, or 
      both may be used. A closing brace ( "}" ) is required to close each
      namespace declaration (items 6b and 7b) before the closing '#endif' 
      for the header file include guard.

3.4.7 Forward declarations needed **must** appear first in the "RAJA" or 
nested namespace before any other statements (item 8).

3.4.8 All class and other type definitions (item 9) **must** appear 
after header file inclusions and forward declarations. A proper class 
prologue **must** appear before the class definition; see Section 4 
for details.


---------------------------------------------------------
3.5 General source file guidelines
---------------------------------------------------------

Consistently-applied conventions and source file organization can help
developer productivity. This section provides general source file 
guidelines. In section :ref:`sourceorg-label`, we describe recommended source
file organization.

3.5.1 Each source file **must** have an associated header file with a matching
name, such as "Foo.hxx" for the source file Foo.cxx".

      **Exceptions:** Test files may not require headers.

3.5.2 Unnecessary header files **should not** be included in source files (i.e.,headers not needed to compile the file standalone).

      Such header file inclusions introduce spurious file dependencies, which
      may increases compilation time unnecessarily.

3.5.3 The order of routines implemented in a source file **should** match the 
order in which they appear in the associated header file.

      This makes the methods easier to locate and compare with documentation
      in the header file.

3.5.4 Each function implementation in a source file **should** be documented according to the guidelines in Section 4.


.. _sourceorg-label:

---------------------------------------------------------
3.6 Source file (\*.cxx extension) content organization
---------------------------------------------------------

Content **must** be organized consistently in all source files. The file 
layout described here is recommended. The following summary uses numbers 
and text to illustrate the basic structure. Details about individual items 
are contained in the guidelines after the summary.

.. code-block:: cpp

   // (1) Doxygen file prologue

   // (2) RAJA copyright and release statement

   // (3) Header file inclusions (only those that are NECESSARY)
   #include "..."

   // (4a) RAJA namespace declaration
   namespace RAJA {

   // (5a) RAJA internal namespace (if used); e.g.,
   namespace awesome {

   // (6) Initialization of static variables and data members, if any; e.g.,
   Foo* MyClass::s_shared_foo = 0;

   // (7) Implementation of static class member functions, if any

   // (8) Implementation of non-static class members and other methods

   // (5b) Internal namespace closing brace (if needed)
   } // awesome namespace closing brace

   // (4b) RAJA namespace closing brace
   } // asctoolkit namespace closing brace

The numbered items in the summary above are referred to in the following
guidelines.

3.6.1 Each source file **must** begin with a Doxygen file prologue (item 1).

      See Section 4 for details.

3.6.2 Each source file **must** contain a comment section that includes the
      RAJA copyright and release statement (item 3).

      See Section 4 for details.

3.6.3 All necessary header file include statements (item 2) **must**
      appear immediately after the copyright and release statement and 
      before any implementation statements in the file.

3.6.4 All contents in a source file **must** follow the same namespace 
inclusion pattern as its corresponding header file (see item 3.4.6).

      Either the main "RAJA" namespace (item 4a) or internal namespace 
      (item 5a) may be used, or both may be used. A closing brace ( "}" ) 
      is required to close each namespace declaration (items 4b and 5b).

3.6.5 When used, static variables and class data members **must** be 
      initialized explicitly in the class source file before any method
      implementations (item 6).




---------------------------------------------------------
3.7 Scope
---------------------------------------------------------

3.7.1 All RAJA code **must** be included in the RAJA namespace.

3.7.2 When appropriate, RAJA code should be included in a nested namespace
with the RAJA namespace.

3.7.3 The 'using directive' **must not** be used in any header file.

      Applying this directive in a header file leverages a bad decision to
      circumvent the namespace across every file that directly or indirectly
      includes that header file. Note that this guideline implies that each
      type name appearing in a header file **must be fully-qualified** (i.e.,
      using the namespace identifier and scope operator) if it resides in a
      different namespace than the contents of the file.

3.7.4 The 'using directive' **may** be used in a source file to avoid using a 
fully-qualified type name at each declaration. Using directives **must** appear
after all "#include" directives in a source file.

3.7.5 When only parts of a namespace are used in an implementation file, only 
those parts **should** be included with a using directive instead of the 
entire namespace contents.

      For example, if you only need the standard library vector container form
      the "std" namespace, it is preferable to use::

         using std::vector;

      rather than::

         using namespace std;

3.7.6 Non-member functions that are meant to be used only in a single source 
file **should** be placed in the unnamed namespace to limit their scope to 
that file.

      This guarantees link-time name conflicts will not occur. For example::

         namespace {
            void myInternalFunction();
         }

3.7.7 Nested classes **should** be private unless they are part of the 
enclosing class interface.

      For example::

         class Outer
         {
            // ...
         private:
            class Inner
            {
               // ...
            };
         };

      When only the enclosing class uses a nested class, making it private
      does not pollute the outer scope needlessly. Furthermore, nested classes
      may be forward declared within the enclosing class definition and then
      defined in the implementation file for the enclosing class. For example::

         class Outer
         {
            class Inner; // forward declaration

            // use name 'Inner' in Outer class definition
         };

         // In Outer.cxx implementation file...
         class Outer::Inner
         {
            // Inner class definition
         }

      This makes it clear that the nested class is only needed in the
      implementation and does not clutter the class definition.

3.7.8 Local variables **should** be declared in the narrowest scope possible 
and as close to first use as possible.

      Minimizing variable scope makes source code easier to comprehend and
      may have performance and other benefits. For example, declaring a loop 
      index inside a for-loop statement such as::

         for (int ii = 0; ...) {

      is preferable to::

         int ii;
         ...
         for (ii = 0; ...) {

      **Exception:** When a local variable is an object, its constructor and
      destructor may be invoked every time a scope (such as a loop) is entered
      and exited, respectively. Thus, instead of this::

         for (int ii = 0; ii < 1000000; ++ii) {
            Foo f;
            f.doSomethingCool(ii);
         }

      it may be more efficient to do this::

         Foo f;
         for (int ii = 0; ii < 1000000; ++ii) {
            f.doSomethingCool(ii);
         }

3.7.9 Static or global variables of class type **must not** be used.

      Due to indeterminate order of construction, their use may cause bugs
      that are very hard to find. Static or global variables that are pointers
      to class types **may** be used and must be initialized properly in a
      single source file.

3.7.10 A reference to any item in the global namespace (which should be rare if needed at all) **should** use the scope operator ("::") to make this clear.

      For example::

         int local_val = ::global_val;


========================================
4 Code Documentation
========================================

This section contains content and formatting guidelines for the code
documentation items mentioned in earlier sections. The aims of these 
guidelines are to:

   * Document files, data types, functions, etc. consistently.
   * Promote good documentation practices so that essential information is 
     presented clearly and lucidly, and which do not over-burden developers.
   * Generate source code documentation using the Doxygen system.


-----------------------------------------
4.1 General documentation considerations
-----------------------------------------

4.1.1 New source code **must** be documented following the guidelines in this 
section. Documentation of existing code **should** be modified to conform to 
these guidelines when appropriate. 

      Documentation of existing code **should** be changed when significant code
      modifications are made (i.e., beyond bug fixes and small changes) and 
      when existing documentation is inadequate.

4.1.2 All header and source files **should** have comments necessary to make 
the code easy to understand. However, extraneous comments (e.g., documenting 
"the obvious") **should** be avoided.

      Code that has clear, descriptive names (functions, variables, etc.) and 
      clear logical structure is preferable to code that relies on a lot of 
      comments for understanding. To be useful, comments must be understood by 
      others and kept current with the actrual code. Generally, maintenance 
      and understanding are better served by rewriting tricky, unclear code 
      than by adding comments to it.

4.1.3 End-of-line comments **should** not be used to document code logic, 
since they tend to be less visible than other comment forms and may be 
difficult to format cleanly. 

      Short end-of-line comments **may** be useful for labeling closing braces 
      associated with nested loops, conditionals, for scope in general, and 
      for documenting local variable declarations.

4.1.4 All comments, except end-of-line comments, **should** be indented to 
match the indentation of the code they are documenting. Multiple line comment 
blocks **should** be aligned vertically on the left.

4.1.5 To make comment text clear and reduce confusion, code comments 
**should** be written in grammatically-correct complete sentences or 
easily understood sentence fragments.

4.1.6 Comments **should** be clearly delimited from executable code with blank 
lines and "blocking characters" (see examples below) to make them stand out 
and, thus, improve the chances they will be read.

4.1.7 Blank lines, indentation, and vertical alignment **should** be used in 
comment blocks to enhance readability, emphasize important information, etc.


--------------------------------------------------------------------
4.2 General Doxygen usage guidelines and summary of common commands
--------------------------------------------------------------------

The Doxygen code documentation system uses C or C++ style comment sections 
with special markings and Doxygen-specific commands to extract documentation 
from source and header files. Although Doxygen provides many sophisticated 
documentation capabilities and can generate a source code manual in a variety 
of formats such as LaTeX, PDF, and HTML, these guidelines address only a small 
subset of Doxygen syntax. The goal of adhering to a small, simple set of 
documentation commands is that developers will be encouraged to build useful 
documentation when they are writing code.

4.2.1 Doxygen comment blocks for **may** use either JavaDoc, Qt style, or one 
of the C++ comment forms described below.

      JavaDoc style comments consist of a C-style comment block starting with
      two \*'s, like this::

         /**
          * ...comment text...
          */

      Qt style comments add an exclamation mark (!) after the opening of a
      C-style comment block,like this::

         /*!
          * ...comment text...
          */

      For JavaDoc or Qt style comments, the asterisk characters ("\*") on
      intermediate lines are optional, but encouraged.

      C++ comment forms use either a block of at least two C++ comment lines, 
      where each line starts with an additional slash::

         ///
         /// ...comment text...
         ///

      or an exclamation mark::

         //!
         //! ...comment text...
         //!

4.2.2 Whichever Doxygen comment block style is used, it **must** be the same 
within a file.

4.2.3 When adding documentation to existing code, the new comments **must** 
use the same comment forms as the exising documentation.

4.2.4 Most Doxygen comments **should** appear immediately before the items they describe. 

      **Exception:** Inline oxygen comments for class/struct data members, 
      enum values, function arguments, etc. **must** appear after the item 
      **on the same line** and **must** use the following syntax::

          /*!< ...comment text... */

      Note that the "<" character must appear immediately after the opening of
      the Doxygen comment (with no space before). This tells Doxygen that the
      comment applies to the item immediately preceding the comment. See
      examples in later sections.

4.2.5 A "brief" description **should** be provided in the Doxygen comment 
section for each of the following items: 

      * A type definition (i.e., class, struct, typedef, enum, etc.) 
      * A macro definition
      * A struct field or C++ class data member
      * A C++ class member function declaration (in the header file class 
        definition) 
      * An unbound function signature (in a header file)
      * A function implementation (when there is no description in the 
        associated header file)

      A brief comment **should** be a concise statement of purpose for an item 
      (usually no more than one line) and must start with the Doxygen command 
      "\\brief" (or "@brief").

      The Doxygen system interprets each comment as either "brief" or 
      "detailed". Brief comments appear in summary sections of the generated 
      documentation. They are typically seen before detailed comments when 
      scanning the documentation; thus good brief comments make it easier to 
      navigate a source code manual.

4.2.6 Important information of a more lengthy nature (e.g., spanning multiple 
lines) **should** be provided for files, major data types and definitions, 
functions, etc. when needed. A detailed comment **must** be separated from a 
brief comment with a blank line.

4.2.7 Summary of commonly used Doxygen commands

This Section provides an overview of commonly used Doxygen commands.
Please see the Doxygen documentation cited in the references at the end of 
these guidelines for more details and information about other commands 
that you may find useful.

Note that to be processed properly, Doxygen commands **must** be preceded with 
either "\\" or "\@" character. For brevity, we use "\\" for all commands 
described here.

   * **\\brief** The "brief" command is used to begin a brief description of 
     a documented item. The brief description ends at the next blank line.
   * **\\file** The "file" command is used to document a file. Doxygen requires
     that to document any global item (function, typedef, enum, etc.), the file
     in which it is defined must be documents. 
   * **\\if** and **\\endif** The "if" command, followed by a label, defines 
     the start of a conditional documentation section. The section ends with a
     matching "endif" command. Conditionals are typically used to 
     enable/disable documentation sections. For example, this may be useful if
     a project wants to provide documentation of all private class members 
     for developer documentation, but wnats to hide private members in 
     documentation for users. Conditional sections are disabled by default 
     and must be explicitly enabled in the doxygen configuration file. 
     Conditional blocks can be nested; nested sections are only enabled if 
     all enclosing sections are. The "\\elseif" command is also available to 
     provide more sophisticated control of conditional documentation.
   * **\\name** The "name" command, followed by a name containing no blank 
     spaces, is used to define a name that can be referred to elsewhere 
     in the documentation (via a link).
   * **\\param** The "param" command documents a function parameter/argument.
     It is followed by the parameter name and description. The "\\param" 
     command can be given an optional attribute to indicate usage of the 
     function argument; possible values are "[in]", "[out]", and "[in,out]".
   * **\\return** The "return" command is used to describe the return value 
     of a function.
   * **\\sa** The "sa" command (i.e., "see also") is used to refer (and 
     provide a link to) another documented item. It is followed by the target 
     of the reference (e.g., class/struct name, function name, documentation 
     page, etc.).
   * **\@{** and **\@}**  These two-character sequences begin and end a 
     grouping of documented items. Optionally, the group can be given a name 
     using the "name" command. Groups are useful for providing additional 
     organization in the documentation, and also when several items can be 
     documented with a single description (e.g., a set of simple, related 
     functions). 

   * **\\verbatim, \\endverbatim** The "verbatim/endverbatim" commands are 
     used to start/stop a block of text that is to appear exactly as it is 
     typed, without additional formatting, in the generated documentation.

   * **-** and **-#** The "-" and "-#" symbols begin an item in a bulleted 
     list or numbered list, respectively. In either case, the item ends at 
     the next blank line or next item.

   * **\\b** and **\\e** These symbols are used to make the next word bold or 
     emphasized/italicized, respectively, in the generated documentation.
   

--------------------------------------------------------------------
4.3 RAJA copyright and release statement
--------------------------------------------------------------------

4.3.1 Each file **must** contain a comment section that includes the RAJA 
release information (using whichever comment characters are appropriate for the language the file is written in). In the interest of brevity, the complete
release statement is summarized here to show the essential information. It 
can be found in any of the RAJA files.

.. code-block:: cpp

   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
   // Copyright (c) 2016, Lawrence Livermore National Security, LLC.
   // 
   // Produced at the Lawrence Livermore National Laboratory.
   //
   // LLNL-CODE-689114
   //
   // All rights reserved.
   //
   // This file is part of RAJA.
   //
   // For additional details, please also read raja/README-license.txt.
   //
   // ...
   //
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


--------------------------------------------------------------------
4.4 File documentation
--------------------------------------------------------------------

4.3.1 Each header files that declares unbound functions, defines enums, typedefs, etc. **must** have a Doxygen file prologue similar to the following:

.. code-block:: cpp

   /*!
    ***************************************************************************
    *
    * \file ...optional name of file...
    *
    * \brief A brief statement describing the file contents/purpose. (optional)
    *
    * Optional detailed explanatory notes about the file.
    *
    * \author Name of file author (optional)
    *
    ****************************************************************************
    */

4.3.2 The Doxygen command "\\file" **must** appear first in the file prologue.

      The "\\file" command identifies the comment section as documentation 
      for the file. Doxygen requires that the file itself must be documented 
      for documentation to be generated for any global item (global function, 
      typedef, enum, etc.) defined in the file.

      The file name may include (part of) the path if the file name is not 
      unique. If the file name is omitted on the line after the "\\file" 
      command, then any documentation in the comment block will belong to 
      the file in which it is located instead of the summary documentation 
      in the listing of documented files.

4.3.3 A brief statement of purpose for the file **should** appear as the first comment after the file. If included, the brief statement, **must** be preceded by the "\\brief" command.

      Brief documentation statements are often helpful to those scanning the 
      documentation.

4.3.4 Any detailed notes about the file **may** be included after the brief comment. If this is done, the detailed comments **must** be separated from the brief statement by a blank line.

4.3.4 The name of the original author of the file **may** be entered after the file notes. If the author's name is included, it **must** be preceded by the "\\author" command.


--------------------------------------------------------------------
4.5 Type documentation
--------------------------------------------------------------------

4.5.1 Each type definition (i.e., class, struct, enum, typedef, etc.) and macro definition appearing in a header file **must** have a Doxygen type definition comment prologue immediately before it. For example

.. code-block:: cpp

   /*!
    ****************************************************************************
    *
    * \brief A brief statement of purpose of the type or macro.
    *
    * Optional detailed information that is helpful in understanding the
    * purpose, usage, etc. of the type/macro ...
    *
    * \sa optional cross-reference to other types, functions, etc...
    * \sa etc...
    *
    * \warning This class is only partially functional.
    *
    ****************************************************************************
    */

Note that Doxygen requires that a compound entity, such as a class, struct, 
etc. must be documented in order to document any of its members.

4.5.2 A brief statement describing the type **must** appear as the first text comment using the Doxygen command "\\brief".

4.5.3 Important details about the item **should** be included after the brief comment and, if included, **must** be separated from the brief comment by a blank line.

4.5.4 Cross-references to other items, such as relevant major types, important functions, etc., **should** be included at the end of the prologue to enhance the navigability of the Doxygen documentation. 

      The Doxygen command "\\sa" (for "see also") **should** appear before each
      such cross-reference so that links are generated in the documentation.

4.5.6 Caveats or limitations about the documented type **should** be noted using the "\\warning" Doxygen command as shown above.


--------------------------------------------------------------------
4.6 Data member documentation
--------------------------------------------------------------------

4.6.1 Each struct field, C++ class data member, etc. **should** have a descriptive comment indicating its purpose. 

     This comment may as appear as a prologue before the item, such as::

        /*!
         *
         * \brief Brief statement of purpose of data member m_mode.
         *
         * Optional detailed information about m_mode...
         */
        int m_mode;

     or, it may appear after the item as an inline comment such as::

        int m_mode; /*!< \brief Brief statement of purpose of m_mode... */

4.6.2 Regardless of which documentation form is used, a brief description of purpose of the definition **must** be included using the Doxygen command "\\brief".

4.6.3 When documenting a data item inline (as in the second example above), the comment must follow the item on the same line.

     The form of an inline Doxygen comment is::

         /*!< \brief ...comment text... */

     Note that the "<" character must be included immediately after the start 
     of the Doxygen comment form (with no space between). This tells Doxygen 
     that the comment corresponds to the item immediately preceding it.

4.6.4 Any detailed notes about an item, if included, **must** appear after the brief comment and be separated from the brief comment with a blank line. 

4.6.5 When a detailed comment is provided, or the brief statement requires more than one line, the prologue comment form **should** be used instead of the inline form to make the documentation easier to read.

4.6.6 If the names of data members are sufficiently clear that their meaning and purpose are obvious to other developers (which should be determined in a code review), then the members **may** be grouped together and documented with a single descriptive comment.

      An example of Doxygen syntax for such a grouping is::

         //@{
         //!  @name Data member description...

         int m_member1;
         int m_member2;
         ...
         //@}


--------------------------------------------------------------------
4.7 Function documentation
--------------------------------------------------------------------

4.7.1 Each unbound functions **should** be be documented with a function prologue in the header file where its prototype appears or in a source file immediately preceding its implementation.

4.7.2 Since C++ class member member functions define the class interface, they **should** be documented with a function prologue immediately preceding their declaration in the class definition.

The following examples show two function prologue variations that may 
be used to document a method in a class definition. The first shows how
to document the function arguments in the function prologue.

.. code-block:: cpp

      /*!
       *************************************************************************
       *
       * \brief Initialize a Foo object to given operation mode.
       *
       * The "read" mode means one thing, while "write" mode means another.
       *
       * \return bool indicating success or failure of initialization.
       *              Success returns true, failure returns false.
       *
       * \param[in] mode OpMode enum value specifying initialization mode.
       *                 ReadMode and WriteMode are valid options.
       *                 Any other value generates a warning message and the
       *                 failure value ("false") is returned.
       *
       *************************************************************************
       */
       bool initMode(OpMode mode);

The second example shows how to document the function arguments inline.

.. code-block:: cpp

      /*!
       ************************************************************************
       *
       * @brief Initialize a Foo object to given operation mode.
       *
       * The "read" mode means one thing, while "write" mode means another.
       *
       * @return bool value indicating success or failure of initialization.
       *             Success returns true, failure returns false.
       *
       *************************************************************************
       */
       bool initMode(OpMode mode /*!< [in] ReadMode, WriteMode are valid options */ );

Note that the first example uses the "\\" character to identify Doxygen 
commands; the second uses "@". Also, the "<" character must appear immediately 
after the start of the Doxygen comment form (with no space between). This 
tells Doxygen that the comment corresponds to the item immediately preceding it.

4.7.3 A brief statement of purpose for a function must appear as the first text comment after the Doxygen command "\\brief" (or "@brief"). 

4.7.4 Any detailed notes about a function, when included, **must** appear after the brief comment and **must** be separated from the brief comment by a blank line.

4.7.4 If the function has a non-void return type, the return value **should** be documented in the prologue using the Doxygen command "\return" (or "@return") preceding a description of the return value. 

      Functions with "void" return type and C++ class constructors and 
      destructors **should not** have such documentation.

4.7.5 Function arguments **should** be documented in the function prologue or inline (as shown above) when the intent or usage of the arguments is not obvious. 

      The inline form of the comment may be preferable when the argument 
      documentation is short. When a longer description is provided (such as 
      when noting the range of valid values, error conditions, etc.) the 
      description **should** be placed within the function prologue for 
      readability. However, the two alternatives for documenting function 
      arguments **must not** be mixed within the documentation of a single 
      function to reduce confusion. 

      In any case, superfluous documentation should be avoided. For example, 
      when there are one or two arguments and their meaning is obvious from 
      their names or the description of the function, providing no comments is 
      better than cluttering the code by documenting the obvious. Comments 
      that impart no useful information are distracting and less useful than 
      no comment at all.

4.7.6 When a function argument is documented in the prologue comment section, the Doxygen command "\param" **should** appear before the comment as in the first example above.

4.7.7. The "in/out" status of each function argument **should** be documented.

       The Doxygen "\param" command supports this directly by allowing such an
       attribute to be specified as "\param[in]", "\param[out]", or 
       "\param[in,out]". Although the inline comment form does not support 
       this, such a description **should** be included; e.g., by using "[in]", 
       "[out]", or "[in,out]" in the comment.

4.7.8 Short, simple functions (e.g., inline methods) **may** be grouped together and documented with a single descriptive comment when this is sufficient.

      An example of Doxygen syntax for such a grouping is::

         //@{
         //! @name Setters for data members

         void setMember1(int arg1) { m_member1 = arg1; }
         void setMember2(int arg2) { m_member2 = arg2; }

         //@}

4.7.9 Typically, important implementation details about a function **should** be documented in the source file where the function is implemented. 

      Header file documentation **should** include only purpose and usage 
      information germane to an interface. When a function has separate 
      implementation documentation, the comments **must** not contain Doxygen 
      syntax. Using Doygen syntax to document an item in more than one location 
      (e.g., header file and source file) can cause undesired Doxygen 
      formatting issues and potentially confusing documentation.
      

      A member of a class may be documented as follows in the source file 
      for the class as follows::

        /*
         ***********************************************************************
         *
         * Set operation mode for a Foo object.
         *
         * Important detailed information about what the function does...
         *
         ***********************************************************************
         */
         bool Foo::initMode(OpMode mode)
         {
            ...function body...
         }


===========================
5 General Code Development
===========================

This section contains various development guidelines intended to improve code 
readability, correctness, portability, consistency, and robustness.


--------------------------------------------------------------------
5.1 General design and implementation considerations
--------------------------------------------------------------------

5.1.1 Simplicity, clarity, ease of modification and extension **should** always be a main goal when writing new code or changing existing code. 

5.1.2 All designs and implementations **should** be reviewed with other team members and refined based on input from others. 

      This is especially important for designs that are complex or potentially 
      unclear. What cannot be easily understood cannot be changed and 
      maintained with confidence.

5.1.3 Each entity (class, struct, variable, function, etc.) **should** embody one clear, well-defined concept. 

      The responsibilities of an entity may increase as it is used in new and 
      different ways. However, changes that divert it from its original intent 
      **should** be avoided. Also, large, monolithic entities that provide too 
      much functionality or which include too many concepts tend to increase 
      code coupling and complexity and introduce undesirable side effects. 
      Smaller, clearly constrained objects are easier to write, test, maintain,
      and use correctly. Also, small, simple objects tend to get used more often
      and reduce code redundancy. Designs and implementations that are overly 
      complexity should be evaluated by the team and modified appropriately.

5.1.4 Global, complex, or opaque data sharing **should** be avoided. Shared data increases coupling and contention between different parts of a code base, which makes maintenance and modification difficult.

5.1.5 When making substantial modifications or stylistic changes to existing code, an attempt **should** be made to make all other code, for example in a source file, consistent with the changes.


--------------------------------------------------------------------
5.2 Code robustness 
--------------------------------------------------------------------

5.2.1 The "const" qualifier **should** be used for variables and methods when appropriate to clearly indicate usage and to take advantage of compiler-based error-checking. 

      Constant declarations make code safer and less error-prone since they 
      enforce intent at compile time. They also simplify code understanding
      because a constant declaration clearly indicates the fact that the state
      of a variable or object will not change in the scope in which the 
      declaration appears.

5.2.2 Preprocessor macros **should not** be used when there is a better alternative, such as an inline function or a constant variable definition. 

      For example, this::

         const double PI = 3.1415926535897932384626433832;

      is preferable to this::

         #define PI (3.1415926535897932384626433832)

      Macros circumvent the ability of a compiler to enforce beneficial 
      language concepts such as scope and type safety. Macros are also 
      context-specific and can produce errors that cannot be understood 
      easily in a debugger. Macros **should be used only** when they are the 
      best choice for a particular situation.

5.2.3 An enumeration type **should** be used instead of macro definitions or "int" data for sets of related constant values. 

      In C++, enums are distinct types with a compile-time specified set of 
      values. Enumeration values cannot be implicitly cast to integers or 
      vice versa -- a "static_cast" operator must be used to make the 
      conversion explicit. Thus, enums provide type and value safety and 
      scoping benefits.

5.2.4 Hard-coded numerical constants and other "magic numbers" **must not** be used directly in code. When such values are needed, they **should** be declared as named constants to enhance code readability and consistency.

5.2.5 Floating point constants **should** always be written with a decimal point and have at least one digit before and after the decimal point for clarity. 

      For example, use "0.5" instead of ".5" and "1.0" instead of "1" or "1.". 


--------------------------------------------------------------------
5.3 Compilation and portability
--------------------------------------------------------------------

5.3.1 All C-only files **must** contain only standard C99 usage. Use of standard C11 features **must** be agreed upon by the project team and be guarded in the code using the "USE_C11" compiler generated macro constant. 

      Changing this guideline requires full concensus of all team members.

5.3.2 All C++ files **must** contain only standard C++03 usage. Use of standard C++11 or C++14 features **must** be agreed upon by the project team. If C++11 standard features are introduced, they **must** be guarded in the code using the "USE_CXX11" compiler generated macro constant. 

      Changing this guideline requires full concensus of all team members.

5.3.3 Special non-standard language constructs, such as GNU extensions, **must not** be used if they hinder portability.

5.3.4 Excessive use of the preprocessor for conditional compilation at a fine granularity (e.g., selectively including or removing individual source lines) **should** be avoided. 

      While it may seem convenient, this practice typically produces confusing 
      and error-prone code. Often, it is better to refactor the code into 
      separate routines or large code blocks subject to conditional compilation
      where it is obvious. The team **should** establish a policy policy for 
      how this is done.

5.3.5 Developers **should** rely on compile-time and link-time errors to check for code correctness and invariants. 

      Errors that occur at run-time and which depend on specific control flow 
      and program state are inconvenient for users and can be difficult to 
      detect and fix.

5.3.6 Before committing code to the source repository, developers **must** attempt to compile cleanly at the highest warning level with the main compiler(s) supported by the project. All warnings **must** be understood and eliminated if possible (not by reducing the warning level!). 

      Compiler warnings, while seemingly innocuous at times, often indicate 
      problems that do not appear until later or until specific run-time 
      conditions are encountered.


--------------------------------------------------------------------
5.4 Memory management
--------------------------------------------------------------------

5.4.1 Memory **should** be deallocated in the same scope in which it is allocated.

5.4.2 Memory **should** be deallocated as soon as it is no longer needed.

5.4.3 Pointers **should** be set to null explicitly when memory is deallocated. 

      For uniformity across the CS Toolkit and to facilitate C++11 and 
      non-C++11 usage, this should be done using the common macro 
      "ATK\_NULLPTR"; For example:: 

         double* data = new double[10];
         // ...
         delete [ ] data;
         data = ATK_NULLPTR;
  
5.4.4 Data managed exclusively within C++ code **must** be allocated and deallocated using the "new" and "delete" operators. 

      The operator "new" is type-safe, simpler to use, and less error-prone 
      than the "malloc" family of C functions.  C++ new/delete operators 
      **must not** be combined with C malloc/free functions.

5.4.5 Every C++ array deallocation statement **must** include "[ ]" (i.e., "delete[ ]") to avoid memory leaks. 

      The rule of thumb is: when "[ ]" appears in the allocation, then "[ ]" 
      **must** appear in the corresponding deallocation statement.  

5.4.6 Before committing code to the source repository, one **should** use memory-checking tools to verify there are no leaks and other memory misuse.

      When merging to the *develop* or *master* branches, compilation with a 
      variety fo compilers, testing, memory-checking, etc. will be done 
      automatically as part of the *pull request* approval process.  The pull
      request will not be approved until all of these tasks succeed.


--------------------------------------------------------------------
5.5 Function declarations
--------------------------------------------------------------------

5.5.1 Any class member function that does not change a data member of the associated class **must** be declared "const".

5.5.2 Function arguments **must** be ordered the same way for all routines in a project.

      Common conventions are either to put all input arguments first, then 
      outputs, or the other way around. Input and output and outputs 
      **must not** be mixed in a function signature. Parameters that are both 
      input and output can make the best choice unclear. Conventions consistent
      with relatd functions **must** always be followed. When adding new 
      parameters to an existing method, the established ordering convention 
      **must** be followed. Do not just stick new parameters at the end of
      the argument list.

5.5.3 Each function argument that is not a built-in type (i.e., int, double, char, etc.) **should** be passed either by reference or as a pointer to avoid unnecessary copies.

5.5.4 Each function reference or pointer argument that is not changed by the function **must** be declared "const".

5.5.6 Variable argument lists (i.e., using ellipses "...") **should** be avoided. 

      Although this is a common practice in C code, and can be done in C++ code,
      this is typically considered a dangerous carryover from C. Variadic 
      functions are not type-safe and they require tight coupling between 
      caller and callee, and can result in undefined behavior.

5.5.7 Each argument in a function declaration **must** be given a name that exactly matches the function implementation. 

      For example, use::

         void computeSomething(int op_count, int mode);

      not::

         void computeSomething(int, int);


--------------------------------------------------------------------
5.6 Function implementations
--------------------------------------------------------------------

5.6.1 Each function body **should** be a reasonable length to be easily understood and viewed in a text editor. Long, complex routines **should** be refactored into smaller parts when this is reasonable to increase clarity, flexibility, and the potential for code reuse.

5.6.2 Each function **should** have exactly one return point to make control logic clear.

      Functions with multiple return points tend to be a source of errors when 
      modifying code. Such routines can always be refactored to have a single 
      return point by using local scope boolean variables and/or different 
      control logic.

      A function **may** have two return points if the first return statement 
      is associated with error condition check, for example. In this case, 
      the error check **should** be performed at the start of the function body
      before other statements are reached. For example, the following is a 
      reasonable use of two function return points because the error condition
      check and the return value for successful completion are clearly visible::

         int computeSomething(int in_val)
         {
            if (in_val < 0) { return -1; }

            // ...rest of function implementation...

            return 0;
         }

5.6.3 "Sanity checks" should be performed on values of function arguments (e.g., range checking, null pointer checking, etc.) upon entry to a function. 

      This is an excellent way to provide run-time debugging capabilities in 
      code. Currently, we have a set of *assertion* macros to make syntax
      consistent. When triggered, they can emit a failed boolean expression and
      descriptive message that help to understand the violation. They are 
      active or not based on the compilation mode, either debug (active) or 
      optimized (inactive). For example::

         void doSomething(int in_val, Foo* in_foo)
         {
            ATK_ASSERT_MSG( in_val >= 0, "in_val must be positive or zero" );
            ATK_ASSERT( in_foo != NULL );

            // ...function implementation...
         }  


--------------------------------------------------------------------
5.7 Inline functions
--------------------------------------------------------------------

Function inlining is a compile time operation and the full definition of an 
inline function must be seen wherever it is called. Thus, any function to be
inlined must be implemented in a header file. 

When a function is implemented in a header file, but not declared inline, a 
compiler will choose whether or not to inline the function. Typically, 
a compiler will not inline a function that is too long or too complex (e.g.,
if it contains complicated conditional logic). When a compiler inlines a 
function, it replace the function call with the body of the function. Most
modern ccompilers do a good job of deciding when inlining is a good choice.

**Important notes:**

  * When a function implementation appears in a header file, every file that
    uses that inline method will often also emit a *function version* of the 
    method in the object file (\*.o file). This is needed to properly
    support function pointers.
  * When a function is explicitly declared inline, using the "inline" keyword,
    the compiler still decides whether to inline the function. It is possible to
    specify function attributes and compiler flags that will force a compiler to
    inline a function. Excessive inlining can cause executable code bloat and 
    may make debugging dificult. Thus, care must be used when deciding which 
    functions to explicitly declare inline. 

**When in doubt, don't use the "inline" keyword and let the compiler decide whether to inline a function.**

5.7.1 Simple, short frequently called functions, such as accessors, **should** be implemented inline in header files in most cases.

      **Exception:** Most accessors that return an object by value (i.e., not by
      pointer or a reference) **should not** be inlined. For example::

         clas MyData 
         {
            // ...public interface...
         private:
            // non-trivial private data members
            vector<Foo> m_foovec;
            Bar m_bar;
         };

         class MyClass
         {
         publis:
            //...
            MyData getData() { return m_mydata; } 

         private:
            MyData m_mydata;
         }; 

5.7.2 Class constructors **should not** be inlined. 

      A class constructor implicitly calls the constructors for its base 
      classes and initializes some or all of its data members, potentially 
      calling more constructors. If a constructor is inlined, the construction 
      and initialization needed for its members and bases will appear at every 
      object declaration.

      **Exception:** The only case where it is reasonable to inline a 
      constructor is when it has only POD ("plain old data") mambers, is not a 
      subclass of a base class, and does not explcitly declare a destructor. 
      In this case, a compiler will not even generate a destructor in most 
      cases. For example::

           class MyClass
           {
           public:
              MyClass() : m_data1(0), m_data2(0) { }

              // No destructor declared

              // ...rest of class definition...
           private:
              // class has only POD members
              int m_data1; 
              int m_data2; 
           };

5.7.3 Virtual functions **must not** be inlined due to polymorphism. 

      For example, do not declare a virtual class member function as::

         virtual void foo( ) { }

      In most circumstances, a virtual method cannot be inlined even though it
      would be inlined otherwise (e.g., because it is very short). A compiler
      must do runtime dispatch on a virtual method when it doesn't know the
      complete type at compile time.

      **Exception:** It is safe to define an empty destructor inline in an
      abstract base class with no data members. For example:: 

           class MyBase
           {
           public:
              virtual ~MyBase() {}

              virtual void doSomething(int param1) = 0;

              virtual void doSomethingElse(int param2) = 0;

              // ...

              // ...no data members...
           };


--------------------------------------------------------------------
5.8 Function and operator overloading
--------------------------------------------------------------------

5.8.1 Functions with the same name **must** differ in their argument lists and/or in their "const" attribute. 

      C++ does not allow identically named functions to differ only in their 
      return type since it is always the option of the caller to ignore or use 
      the function return value.

5.8.2 Function overloading **must not** be used to define functions that do conceptually different things. 

      Someone reading declarations of overloaded functions should be able to 
      assume (and rightfully so!) that functions with the same name do 
      something very similar.

5.8.3 If an overloaded virtual method in a base class is overridden in a derived class, all overloaded methods with the same name in the base class **must** be overridden in the derived class. 

      This prevents unexpected behavior when calling such member functions. 
      Remember that when a virtual function is overridden, the overloads of 
      that function in the base class **are not visible** to the derived class.

5.8.4 Operator overloading **must not** be used to be clever to the point of obfuscation and cause others to think too hard about an operation. Specifically, an overloaded operator must preserve "natural" semantics by appealing to common conventions and **must** have meaning similar to non-overloaded operators of the same name.

      Overloading operators can be beneficial, but **should not** be overused 
      or abused. Operator overloading is essentially "syntactic sugar" and an
      overloaded operator is just a function like any other function. An 
      important benefit of overloading is that it often allows more 
      appropriate syntax that more easily communicates the meaning of an 
      operation. The resulting code can be easier to write, maintain, and 
      understand, and it may be more efficient since it may allow the compiler
      to take advantage of longer expressions than it could otherwise.

5.8.5 Both boolean operators "==" and "!=" **should** be implemented if one of them is. 

      For consistency and correctness, the "!=" operator **should** be 
      implemented using the "==" operator implementation. For example::

         bool MyClass::operator!= (const MyClass& rhs)
         {
            return !(this == rhs);
         }

5.8.6 Standard operators, such as "&&", "||", and "," (i.e., comma), **must not** be overloaded.

      The built-in versions are treated specially by the compiler. Thus, 
      programmers cannot implement their full semantics. This can cause
      confusion. For example, the order of operand evaluation cannot be 
      guaranteed when overloading operators "&&" or "||". This may cause
      problems as someone may write code that assumes that evaluation order 
      is the same as the built-in versions.


--------------------------------------------------------------------
5.9 Types
--------------------------------------------------------------------

5.9.1 Behavior **should not** be selected by "switching" on the type of an 
object. 

      Good object-oriented design uses virtual functions (or templates) to 
      decide behavior. Using conditional logic (e.g., in calling code) to
      decide behavior is often unsafe and error-prone, and a clear indication 
      of poor design and improper use of the C++ type system.

5.9.2 The "bool" type **should** be used in C++ code instead of "int" for boolean true/false values.

5.9.3 The "string" type **should** be used in C++ code instead of "char\*". 

      The string type supports and optimizes many character string manipulation
      operations which can be error-prone and less efficient if implemented 
      explicitly using "char\*" and standard C library functions. Note that 
      "string" and "char\*" types are easily interchangeable, which allows C++ 
      string data to be used when interacting with C routines.

5.9.4 Class type variables **should** be defined using direct initialization instead of copy initialization to avoid unwanted and spurious type conversions and constructor calls that may be generated by compilers. 

      For example, use:: 

         std::string name("Bill");

      instead of::

         std::string name = "Bill";

      or::

         std::string name = std::string("Bill");


--------------------------------------------------------------------
5.10 Type casting
--------------------------------------------------------------------

5.10.1 C-style casts **must not** be used in C++ code. 

      All type conversions **must** be done explicitly using the named C++ 
      casting operators; i.e., "static_cast", "const_cast", "dynamic_cast", 
      "reinterpret_cast".

5.10.2 The choice to use the "static_cast" or "dynamic_cast" operator on pointers **must** consider the performance context of the code.

       The "dynamic_cast" operator is a more powerful and safer way to cast 
       pointers. However, in performance critical code, dynamic cast overhead 
       may be unacceptable. Static casts are done at compile time and are 
       essentially free at runtime whereas each dynamic cast may incur hundreds        of cycles of runtime overhead. When this choice is encountered, it may
       be wise to consider other implementation alternatives.

5.10.3 The "const_cast" operator **should** be avoided. 

       Casting away "const-ness" is often a poor programming decision and can 
       introduce errors.

       **Exception:** It may be necessary in some circumstances to cast away 
       const-ness, such as when calling const-incorrect APIs.

5.10.4 The "reinterpret_cast" **must not** be used unless absolutely necessary.

       This operator was designed to perform a low-level reinterpretation of 
       the bit pattern of an operand. This is needed only in special 
       circumstances and circumvents type safety.


--------------------------------------------------------------------
5.11 Templates
--------------------------------------------------------------------

5.11.1 Typically, a class (or function) template **should** be used only when the behavior of the class (or function) is completely independent of the type of the object to which it is applied. 

       Note that class member templates (e.g., member functions that are 
       templates of a class that is not a template) are often useful to 
       reduce code redundancy.

5.11.2 Generic templates that have external linkage **must** be defined in the header file where they are declared since template instantiation is a compile time operation. Thus, implementations of class templates and member templates **must** be placed in the class header file, preferably after the class definition.

5.11.3 Complete specializations of member templates or function templates **must not** appear in a header file. 

       Such methods **are not templates** and may produce link errors if their 
       definitions are seen more than once.


--------------------------------------------------------------------
5.12 Conditional statements and loops
--------------------------------------------------------------------

5.12.1 Curly braces **must** be used in all conditionals, loops, etc. even when the content inside the braces is a "one-liner". 

       This helps prevent coding errors and misinterpretation of intent. 
       For example, use::

          if (done) { ... }

       not::

          if (done) ...

5.12.2 One-liners **may** not be used for "if" conditionals with "else/else if"  clauses when the resulting code is clear. 

       For example, either of the following styles **may** be used::

          if (done) {
             id = 3;
          } else {
             id = 0;
          }

       or::

          if (done) { id = 3; } else { id = 0; }

5.12.3 For clarity, the shortest block of an "if/else" statement **should** come first.

5.12.4 Complex "if/else if" conditionals with many "else if" clauses **should** be avoided.

      Such statements can always be refactored using local boolean variables 
      or "switch" statements. Doing so often makes code easier to read and 
      understand and may improve performance.

5.12.5 An explicit test for zero/nonzero **must** be used in a conditional unless the tested quantity is a bool or a pointer. 

      For example, a conditional based on an integer value should use::

         if (num_lines != 0) {

      not::

         if (num_lines) {

5.12.6 A switch statement **should** use curly braces for each case and use indentation, white space, and comments for readability. Also, each case **must** contain a "break" statement and a "default" case **must** be provided to catch erroneous case values. "Fall through" cases are confusing and error-prone and so **should** be made clear in the code.

      Here is an example illustrating several preferred style practices.

.. code-block:: cpp

         switch (condition) {

            case ABC : {
               ...
               break;
            }

            case DEF :  // fall-through case
            case GHI : {
               ...
            break;
            }

            default : {
            ...
            }

         }

This code example has the following desirable properties:

   * Curly braces are used for the "switch" statement and for each case.
   * Each "case" statement is indented within the "switch" statement.
   * Blank lines are used between different cases.
   * Each case containing executable statements has a "break" statement.
   * Fall-through case is documented.
   * A "default" case is provided to catch erroneous case values.

5.12.7 The "goto" statement **should not** be used. 

      Only if alternatives are considered and determined to be less desirable, 
      should a "goto" even be contemplated.


--------------------------------------------------------------------
5.13 White space
--------------------------------------------------------------------

5.13.1 Blank lines and indentation **should** be used throughout code to enhance readability. 

      Examples of helpful white space include:

         * Between operands and operators in arithmetic expressions.
         * After reserved words, such as "while", "for", "if", "switch", etc. 
           and before the parenthesis or curly brace that follows.
         * After commas separating arguments in functions.
         * After semi-colons in for-loop expressions.
         * Before and after curly braces in almost all cases.

5.13.2 White space **must not** appear between a function name and the opening parenthesis to the argument list.  In particular, if a function call is broken across source lines, the break **must not** come between the function name and the opening parenthesis.

5.13.3 Tabs **must not** be used for indentation since this can be problematic for developers with different text editor settings.


--------------------------------------------------------------------
5.14 Code alignment
--------------------------------------------------------------------

5.14.1 Each argument in a function declaration or implementation **should** appear on its own line for clarity. The first argument **may** appear on the same line as the function name. When function areguments are placed on multiple lines, they **should** be aligned vertically for clarity.

5.14.2 All statements within a function body **should** be indented within the surrounding curly braces.

5.14.3 The start of all source lines in the same scope **should** be aligned vertically, except for continuations of previous lines.

5.14.4 If a source line is broken at a comma or semi-colon, it **must** be broken after the comma or semi-colon, not before. 

      Doing otherwise, produces code that is hard to read and can lead to 
      errors.

5.14.5 If a source line is broken at an arithmetic operator (i.e., , -, etc.), it **should** be broken after the operator, not before. 

      Doing otherwise, yields code that is harder to read and can lead to 
      errors.

5.14.6 Parentheses **should** be used in non-trivial mathematical and logical expressions to clearly indicate structure and intended order of operations and to enhance readability. 

      Do not assume everyone who looks at the code knows all the rules for 
      operator precedence.


===================================================
6 Class Design and Implementation
===================================================

---------------------------------------------------
6.1 C++ class definition structure and guidelines
---------------------------------------------------

This section contains guidelines for structuring a C++ class definition. 
The summary here uses numbers and text to illustrate the basic structure.
Details about individual items follow.

.. code-block:: cpp

   /* (1) Class definition preceded by documentation prologue */

   /*! 
    * \brief ...summary comment text...
    *
    * ...detailed comment text... 
    */
   class MyClass
   {

      /* (2) "friend" declarations (if needed) */

   /* (3) "public" members */
   public:

      /* (3a) static member function declarations (if needed) */

      /* (3b) public member function declarations */

   /* (4) "protected" members (rarely needed) */
   protected:
 
      /* (4a) protected member function declarations (if needed) */

   /* (5) "private members */
   private:

      /* (5a) private static data members (if needed) */

      /* (5b) private member function declarations */

      /* (5c) private data member declarations */

   };

The numbers in parentheses in the following guidelines correspond to the 
numbered items in the preceding summary.

6.1.1 Each class definition **must** be preceded by a Doxygen documentation prologue. 

      See Section 4 for details.

6.1.2 Both the opening curly brace "{" and the closing curly brace "};" for a class definition **must** be on their own source lines and must be aligned vertically with the "class" reserved word.

6.1.3 "Friend" declarations should be needed rarely if at all, but if used, they**must** appear within the body of a class definition before any class member declarations (2).

      Note that placing "friend" declarations before the "public:" keyword makes      them private, which should be the case in most circumstances. 

6.1.4 Class members **must** be declared in the following order: 

      # "public" (item 3 in summary)
      # "protected" (item 4 in summary)
      # "private" (item 5 in summary)

      That is, order members by decreasing scope of audience.

6.1.5 Static class members (methods and data) **must** be used rarely, if at all. In every case, there usage **must** be considered carefully.

      When it is determined that a static member is needed, it **must** appear 
      first in the appropriate member section. Typically, static member 
      functions **should** be "public" (item 3a in summary) and static data
      members **should** be "private" (item 5a in summary).

6.1.6 Within each set of member declarations (i.e., public, protected, private), all function declarations **must** appear before data member declarations (items 3a and 3b, 4a, 5b and 5c in summary).

6.1.7 Class data members **should** be "private" almost always. If "public" or "protected" data members are even considered, this choice **must** be reviewed carefully by other team members.

      Information hiding is an essential aspect of good software engineering 
      and private data is the best means for a class to preserve its 
      invariants. Specifically, a class should maintain control of how object 
      state can be modified to minimize side effects. In addition, restricting
      direct access to class data enforces encapsulation and facilitates 
      design changes through refactoring.

      Note that "public" and "protected" data members are not included in the 
      summary above to reinforce this guideline.

6.1.9  A class constructor that takes a single *non-default* argument, or a single argument that has a *default* value, **must** be declared "explicit". 

       This will prevent compilers from performing unexpected (and, in many
       cases, unwanted!) implicit type conversions. For example::

          class MyClass
          {
          public:
             explicit MyClass(int i, double x = 0.0);
          };

       Note that, without the explicit declaration, an implicit conversion from
       an integer to an object of type MyClass is allowed. For example::

          MyClass mc = 2;

       Clearly, this is confusing. The "explicit" keyword forces the following::

          MyClass mc(2); 

       to get the same result, which is much more clear.

6.1.10 Each class member function that does not change the object state **must** be declared "const". 

       This helps compilers detect usage errors.

6.1.11 Each class member function that returns a class data member that should not be changed by the caller **must** be declared "const" and **must** return the data member as a "const" reference or pointer.

       Often, both "const" and non-"const" versions of member access functions 
       are needed so that callers may declare the variable that holds the 
       return value with the appropriate "const-ness".

6.1.12 If a class contains nested classes or other types, these definitions **should** appear before other class members (i.e., data and functions) within the appropriate section ("public" or "private") of the enclosing class definition.

       See Section 3.8 for further guidance.

6.1.13 Each class member function and data member declaration **must** be properly documented according to the guidelines Section 4.


---------------------------------------------------
6.2 Class member initialization and copying
---------------------------------------------------

6.2.1 Every class data member **must** be initialized (using default values when appropriate) in each class constructor. That is, an initializer/initialization **must** be provided for each class data member so that every object is in a well-defined state upon construction. 

      Generally, this requires a user-defined default constructor when a class 
      has POD members. Do not assume that a compiler-generated default 
      constructor will leave any member variable in a well-defined state.

      **Exception:** A class that has no member variables, including one that 
      is derived from a base class with a default constructor that provides 
      full member initialization does not require a user-defined default 
      constructor since the compiler-generated version will suffice.

6.2.2 Data member initialization **should** be used instead of assignment in constructors, especially for small classes. 

      Initialization prevents needless run-time work and is often faster.

6.2.3 For classes with complex data members, assignment within the body of the constructor **may** be preferable. 

      If the initialization process is sufficiently complex, it **may** be 
      better to perform object initialization in a method that is called 
      after object creation, such as "init()".

6.2.4 When using initialization instead of assignment to set data member values in a constructor, the data members **should** always be initialized in the order in which they appear in the class definition. 

      Compilers adhere to this order regardless of the order that members 
      appear in the class initialization list. So you may as well agree with 
      the compiler rules and avoid potential errors when initialization of 
      one member depends on the state of another.

6.2.5 A constructor **must not** call a virtual function on any data member object since an overridden method defined in a subclass cannot be called until the object is fully constructed. 

      There is no general guarantee that data members are fully-created 
      before a constructor exits.

6.2.6 All memory allocated in a class constructor **must** be de-allocated in the class destructor. 

      Note that the intent of constructors is to acquire resources and the 
      intent of destructors is to free those resources.

6.2.7 A user-supplied implementation of a class copy-assignment operator **should** check for assignment to self, **must** copy all data members from the object passed to operator, and **must** return a reference to "\*this".

      The *copy-and-swap* idiom **should** be used. 

      See Appendix B for a detailed example of how this is done.

6.2.8 All constructors and copy operations for a derived class **must** call thenecessary constructors and copy operations for each of its base classes to insure that each object is properly allocated and initialized.


---------------------------------------------------
6.3 Compiler-generated methods
---------------------------------------------------

The guidelines in this section apply to class methods that may be 
*automatically generated* by a compiler, in particular, the default 
constructor, destructor, copy constructor, and copy-assignment operator.
Similar guidelines apply to the move constructor and move-assignment operator
in C++11. See Section 7 for C++11 guidelines. Also, see Appendix A, which
describes the rules under which a C++ complier will generate class methods
automatically.

6.3.1 Each class **must** follow the *rule of three* which states: if the destructor, copy constructor, or copy-assignment operator is explicitly defined, then the others **must** be defined.

      Compiler-generated and explicit versions of these methods **must not**
      be mixed. If a class requires one of these methods to be implemented, 
      it almost certainly requires all three to be implemented. 

      The reason for this rule is to insure that class resources are managed 
      properly. C++ copies and copy-assigns objects of user-defined types in 
      various situations (e.g., passing/returning by value, manipulating a 
      container, etc.). These special member functions will be called, if 
      accessible. If they are not user-defined, they are implicitly-defined 
      by the compiler.

      The compiler-generated special member functions are often incorrect 
      if a class manages a resource whose handle is an object of 
      non-class type. Consider a class data member which is a raw pointer to an
      an object of a class type. The compiler-generated class destructor will
      not free the member object. Also, the compiler-generated copy constructor       and copy-assignment operator will perform a "shallow copy"; i.e., they 
      will copy the value of the pointer without duplicating the underlying 
      resource.

      Similarly, each class **must** follow the *rule of five* when using 
      C++11 features.  See Section 7 for details.

6.3.2 Classes that manage non-copyable resources through non-copyable handles, such as pointers, **should** declare the compiler-generated methods private and and leave them unimplemented.

      When the intent is that such methods should never be called, this is a 
      good way to help a compiler to catch unintended usage. For example::

	   class MyClass
	   {
	      // ...

	   private:
	      // The following methods are not implemented
	      MyClass();
	      MyClass(const MyClass&);
	      void operator=(const MyClass&);

	      // ...
	   };

      When code does not have private access to the class tries to use such
      a method, a compile-time error will result. If a class does have private
      access and tries to use one of these methods an link-time error will
      result. 

      This is another application of the "rule of three".

      This guideline extends to the additional compiler-generated methods in
      C++11. See Section 7 for details.

6.3.3 The default constructor, copy constructor, destructor, and copy assignment **may** be left undeclared when the compiler-generated versions are appropriate. In this case, the class header file **should** contain comments indicating that the compiler-generated versions of these methods will be used.

      **Exception:** If a class inherits from a base class that declares
      these methods private, the subclass need not declare the methods
      private. However, a comment **should** be provided in the derived
      class stating that the parent class enforces the non-copyable
      properties of the class.

6.3.4 If a class is default-constructable and has POD data members, including raw pointers, the default constructor **must** be defined explicitly and the data members **must** be initialized explicitly. A compiler-generated version of a default constructor will not initialize such members, in general.

6.3.5 By convention, a functor class **should** have a copy constructor and copy-assignment operator. 

      Typically, the compiler-generated versions are sufficient when the class 
      has no state or non-POD data members. Since such classes are usually 
      small and simple, the compiler-generated versions of these methods 
      **may** be used without documenting the use of default value semantics 
      in the functor definition.


---------------------------------------------------
6.4 Inheritance
---------------------------------------------------

6.4.1 Class composition **should** be used instead of inheritance to extend behavior. 

      Looser coupling between objects is typically more flexible and easier 
      to maintain and refactor.

6.4.2 Class hierarchies **should** be designed so that subclasses inherit from abstract interfaces; i.e., pure virtual base classes. 

      Inheritance is often done to reuse code that exists in a base class. 
      However, there are usually better design choices to achieve reuse. 
      Good object-oriented use of inheritance is to reuse existing *calling* 
      code by exploiting base class interfaces using polymorphism. Put another 
      way, "interface inheritance" should be used instead of "implementation 
      inheritance".

6.4.3 Deep inheritance hierarchies; i.e., more than 2 or 3 levels, **should** be avoided.

6.4.4 Multiple inheritance **should** be restricted so that only one base class contains methods that are not "pure virtual"; i.e., adhering to the Java model of inheritance is most effective for avoiding abuse of inheritance.

6.4.5 One **should not** inherit from a class that was not designed to be a base class (e.g., if it does not have a virtual destructor). 

      Doing so is bad practice and can cause problems that may not be reported 
      by a compiler; e.g., hiding base class members. To add functionality, 
      one **should** employ class composition rather than by "tweaking" an 
      existing class.

6.4.6 The destructor of a class that is designed to be a base class **must** be declared "virtual". 

      However, sometimes a destructor should not be declared virtual, such as 
      when deletion through a pointer to a base class object should be 
      disallowed.

6.4.7 "Private" and "protected" inheritance **must not** be used unless you absolutely understand the ramifications of such a choice and are sure that it will not create design and implementation problems. 

      Such a choice **must** be reviewed with team members. There almost 
      always exist better alternatives to avoid these forms of inheritance.

6.4.8 Virtual functions **should** be overridden responsibly. That is, the pre- and post-conditions, default arguments, etc. of the virtual functions should be preserved. 

      Also, the behavior of an overridden virtual function **should not** 
      deviate from the intent of the base class. Remember that derived classes 
      are subsets, not supersets, of their base classes.

6.4.9 A virtual function in a base class **should only** be defined if its behavior is always valid default behavior for *any* derived class.  

6.4.10 Inherited non-virtual methods **must not** be overloaded or hidden.

6.4.11 If a virtual function in a base class is not expected to be overridden in any derived class, then the method **should not** be declared virtual.

6.4.12 If each derived class has to provide specific behavior for a base class virtual function, then it **should** be declared *pure virtual*.

6.4.13 Virtual functions **must not** be called in a class constructor or destructor. Doing so is undefined behavior according to the C++ standard. Even if it seems to work correctly, it is fragile and potentially non-portable.

6.2.14 A constructor for a derived class **must** call the appropriate constructor for each of its base classes as needed to insure that each object is properly allocated and initialized.

6.2.15 Copy operations for a derived class **must** call the appropriate copy operations for each of its base classes as needed to insure that each object is properly allocated and initialized.


===============================================
7 Restrictions on Language Usage and Libraries
===============================================

C++ is a huge language with many advanced and powerful features. To avoid
over-indulgence and obfuscation, we would like to avoid C++ feature bloat.
By constraining, or even banning, the use of certain language features and
libraries we hope to keep code simple, portable, and avoid errors and 
problems that may occur when language features are not completely 
understood or used consistently.  This section lists such restrictions and 
explains why use of certain features is constrained or restricted.


------------------------
7.1 C++11 and beyond
------------------------

Applications that use the CS Toolkit will rely on non-C++11 compilers for 
our current generation of computing platforms, and possibly beyond, so we
must be able to compile and run our code with those compilers.

C++11 may be used in the CS Toolkit in limited ways as described in this 
section. Any other usage must be carefully reviewed and approved by all 
team members.

7.1.1 All C++11 usage **must** be guarded using the macro constant "USE_CXX11" so that it can be compiled out of the code when necessary. 

   For example, suppose you have a class that you want to support *move* 
   semantics when available (i.e., when using a C++11-compilant compiler) 
   and fall back to copy semantics otherwise:

.. code-block:: cpp

   class MyClass
   {
   public:

      /// Default constructor.
      MyClass();

      /// Destructor.
      ~MyClass();

      /// Copy constructor.
      MyClass(const MyClass& other);

      /// Copy-assignment operator.
      MyClass& operator=(const MyClass& rhs);

   #if defined(USE_CXX11)
      /// Move constructor.
      MyClass(MyClass&& other);

      /// Move-assignment operator.
      MyClass& operator=(MyClass&& rhs);
   #endif 

      // other methods...

   private:
      // data members...
   }; 

7.1.2 Whenever C++11 features are used, an alternative implementation **must** be provided that conforms to the 2003 C++ standard.

      Applications that use the CS Toolkit will expect the code able to compile
      and run with full functionality on all platforms they use. 

7.1.3 C++14 features **must not** be used due to substantially incomplete compiler support on the platforms we care most about.


**WE NEED TO WORK ON THIS SECTION**


------------------------
7.2 Boost
------------------------

The Boost C++ libraries are generally high quality and provide many powerful
and useful capabilities not found in the core C++ language. Indeed, some Boost
libraries eventually make their way into the C++ standard.
 
Some LLNL codes have used Boost successfully for many years. However, 
version inconsistencies (e.g., changes from one version of Boost to the next or 
two codes using different incompatible versions that need to be compiled
and linked into the same executable) and compiler portability have presented 
problems in the past. To avoid increasing the maintenace burden for 
applications that use the CS Toolkit, we restrict Boost usage in the CS 
Toolkit as described in this section. Any other usage must be carefully 
reviewed and approved by all team members.

7.2.1 All CS Toolkit components **must** use the same version of Boost that is maintained for the Toolkit.

7.2.2 Boost libraries that require compilation **must not** be used. That is, only those libraries that provide header files **may** be used.

7.2.3 Boost usage **must not** be exposed through any public interface in the CS Toolkit. 


.. _codingrefs-label:

======================================
References and useful resources
======================================

Most of the guidelines here were gathered from the following list sources. 
The list contains a variety of useful resources for programming in C++
beyond what is presented in these guidelines.

#. *The Chromium Projects: C++ Dos and Don'ts*. https://www.chromium.org/developers/coding-style/cpp-dos-and-donts

#. Dewhurst, S., *C++ Gotchas: Avoiding Common Problems in Coding and Design*, Addison-Wesley, 2003.

#. Dewhurst S., *C++ Common Knowledge: Essential Intermediate Programming*, Addison-Wesley, 2005.

#. *Doxygen manual*, http://www.stack.nl/~dimitri/doxygen/manual/index.html

#. *Google C++ Style Guide*, https://google-styleguide.googlecode.com/svn/trunk/cppguide.html

#. *ISO/IEC 14882:2011 C++ Programming Language Standard*.

#. Josuttis, N., *The C++ Standard Library: A Tutorial and Reference, Second Edition*, Addison-Wesley, 2012.

#. Meyers, S., *More Effective C++: 35 New Ways to Improve Your Programs and Designs*, Addison-Wesley, 1996.

#. Meyers, S., *Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library*, Addison-Wesley, 2001.

#. Meyers, S., *Effective C++: 55 Specific Ways to Improve Your Programs and Designs (3rd Edition)*, Addison-Wesley, 2005.

#. Meyers, S., *Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 and C++14*, O'Reilly.

#. Programming Research Ltd., *High-integrity C++ Coding Standard, Version 4.0*, 2013.

#. Sutter, H. and A. Alexandrescu, *C++ Coding Standards: 101 Rules, Guidelines, and Best Practices*, Addison-Wesley, 2005.
