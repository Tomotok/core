Graphical User Interface
========================

The graphical user interface can be used for displaying and post-processing of tomography results. An application for setting up and computing inversions is no longer supported.

It is based on the tkinter library and designed as a multiwindow application. The main window spawns windows for displaying and/or postprocessing of tomography result. 

GUI functionality is subject to ongoing change and this documentation can be outdated, but the main principle of multiwindow application should not change.

GUI consist of widgets that are controlled by objects called Applications. Currently, there are two types, one that tries to load from a predefined local database the other allow user to specify a location of hdf file. They might need to redirect standard output and should be launched using special convenience function launcher.

The application consist of menu widgets and after data were loaded, it can be used to creates other widget windows for displaying the results.

Widgets could be categorized according to they purpose. There are general purpose widgets that enhance the application, widgets for data visualisation and also widgets that apart from visualisation can also process the data. For description of the displaying and processing methods for multiple reconstructions see [G1]_.

.. rubric:: References

.. [G1] J. Svoboda, et al. "Comparative analysis and new post-processing methods for plasma tomography at tokamaks." Journal of Instrumentation 14.11 (2019): C11001.
