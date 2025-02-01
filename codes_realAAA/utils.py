################################################################################
################################## - utils.py - ################################
# This file is not intended to be run directly, as it is a helper for the main
# script execution. It is responsible for containing useful technical
# functions, like the ones used to generate the synthetic inlet and
# observation velocity expressions. However, most of them belongs to
# synthetic data code group and are obsolete here.
################################################################################
################################################################################

#-----------------------------# IMPORT LIBRARIES #-----------------------------#
import time
from mshr import *
from dolfin import *
from dolfin_adjoint import *
import json
import sys
import logging
import os, errno
from munch import DefaultMunch
from os.path import join
from datetime import datetime

#-----------------------------# DEFINITION OF BOUNDARIES #-----------------------------#
# OBSOLETE:
# Defines boundary conditions for 2D synthetic cases.
# Used only in synthetic data codes when BCs and observations were artificially generated.
class Inlet2D(SubDomain):
    def inside(self, x, on_boundary):
        # Checks if the point x is on the leftmost boundary (x = 0)
        return on_boundary and near(x[0], 0)

class Outlet2D(SubDomain):
    def inside(self, x, on_boundary):
        # Checks if the point x is on the rightmost boundary (x = 200)
        return on_boundary and near(x[0], 200)

class Walls2D(SubDomain):
    def inside(self, x, on_boundary):
        # Checks if the point x is on the top or bottom boundary (y = 0 or y = 41)
        return on_boundary and near(x[1], 0) and near(x[1], 41)

#-----------------------------# READING .json CONFIGURATION FILE #-----------------------------#
def load_conf_file(config_file):
   """
    Loads a JSON configuration file and converts it into an accessible object.

    Parameters:
    - config_file (str): Path to the JSON configuration file.

    Returns:
    - cf (DefaultMunch): Object that allows accessing JSON keys as attributes.
   """
   # Check if the provided file has a .json extension
   if 'json' in config_file:
       with open(config_file, 'r') as f:
           cfg = json.load(f)
       Conf = object()
       # Convert dictionary into an attribute-accessible object
       cf = DefaultMunch.fromDict(cfg, Conf)
       return cf

#-----------------------------# BOUNDARY FUNCTION FOR INLET VELOCITY - 2D TIME VARIANT #-----------------------------#
# OBSOLETE:
# Defines boundary conditions for 2D synthetic cases.
# Used only in synthetic data codes when BCs and observations were artificially generated.
class BoundaryFunction2D(UserExpression):
    def __init__(self, t, Umax_g, profile, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.Umax_g = Umax_g
        self.profile = profile

    def eval(self, values, x):
        if self.profile == "Plug":
            # Plug flow profile: uniform velocity with a sinusoidal time variation.
            U = 4 * self.Umax_g * x[1] * (41 - x[1]) / pow(41, 2) * sin(pi * 0.05 / 0.3)
            values[0] = U  # Velocity in x-direction
            values[1] = 0  # Velocity in y-direction
        elif self.profile == "Parabolic":
            # Parabolic profile: velocity distribution follows a parabolic shape.
            # Initial systolic phase with a parabolic velocity profile
            if self.t < 0.25:
                U = 4 * self.Umax_g * x[1] * (41 - x[1]) / pow(41, 2) * sin(pi * self.t / 0.3)
            # Diastolic phase: flow is forced into a plug profile
            elif 0.25 <= self.t < 0.85 or 1.05 <= self.t < 1.65 or self.t >= 1.85:
                U = 4 * self.Umax_g * x[1] * (41 - x[1]) / pow(41, 2) * sin(pi * 0.05 / 0.3)
            # Transitioning phase back to a parabolic profile
            elif 0.85 <= self.t < 1.05:
                U = 4 * self.Umax_g * x[1] * (41 - x[1]) / pow(41, 2) * sin(pi * (self.t - 0.8) / 0.3)
            # Parabolic transition
            elif 1.65 <= self.t < 1.85:
                U = 4 * self.Umax_g * x[1] * (41 - x[1]) / pow(41, 2) * sin(pi * (self.t - 1.6) / 0.3)
            values[0] = U  # Velocity in x-direction
            values[1] = 0  # Velocity in y-direction

    def value_shape(self):
        return (2,)  # The function returns a 2D vector

#-----------------------------# BOUNDARY FUNCTION FOR INLET VELOCITY - 2D STATIONARY #-----------------------------#
# OBSOLETE:
# Defines boundary conditions for 2D synthetic cases.
# Used only in synthetic data codes when BCs and observations were artificially generated.
class StationaryBoundaryFunction2D(UserExpression):
    def __init__(self, Umax, profile, **kwargs):
        super().__init__(**kwargs)
        self.Umax = Umax
        self.profile = profile

    def eval(self, values, x):
        if self.profile == "Plug":
            # Uniform (plug) velocity profile with constant velocity
            U = self.Umax  # Uniform velocity profile
        if self.profile == "Parabolic":
            # Parabolic velocity profile (maximum velocity at the center and zero at the boundary)
            U = 4 * self.Umax * x[1] * (41 - x[1]) / pow(41, 2)  # Parabolic velocity profile
        values[0] = U  # Velocity in x-direction
        values[1] = 0  # Velocity in y-direction

    def value_shape(self):
        return (2,)

#-----------------------------# BOUNDARY FUNCTION FOR INLET VELOCITY - 3D TIME VARIANT #-----------------------------#
# OBSOLETE:
# Defines boundary conditions for 3D synthetic cases.
# Used only in synthetic data codes when BCs and observations were artificially generated.
class BoundaryFunction3D(UserExpression):
    def __init__(self, t, radius, Umax, profile, **kwargs):
        super().__init__(**kwargs)
        self.t = t  # Time variable
        self.radius = radius  # Radius of the inlet section
        self.Umax = Umax  # Maximum velocity
        self.profile = profile  # Velocity profile type

    def eval(self, values, x):
        if self.profile == "Plug":
            # Plug flow profile: uniform velocity in the radial direction
            U = self.Umax * (1.0 - (x[0]**2 + x[1]**2) / self.radius**2)
        if self.profile == "Parabolic":
            # Parabolic velocity profile with time variation
            if self.t < 0.25:
                # Systolic phase with parabolic profile and sinusoidal time dependence
                U = self.Umax * (1.0 - (x[0]**2 + x[1]**2) / self.radius**2) * sin(pi * self.t / 0.3)
            else:
                # Diastolic phase with plug profile and sinusoidal time dependence
                U = self.Umax * (1.0 - (x[0]**2 + x[1]**2) / self.radius**2) * sin(pi * 0.05 / 0.3)
        values[0] = 0  # Velocity in x-direction
        values[1] = 0  # Velocity in y-direction
        values[2] = U  # Velocity in z-direction

    def value_shape(self):
        return (3,)


class _AnsiColorizer(object):
   """
   A colorizer is an object that loosely wraps around a stream, allowing
   callers to write text to the stream in a particular color.

   Colorizer classes must implement C{supported()} and C{write(text, color)}.
   """
   _colors = dict(black=30, red=31, green=32, yellow=33,
                  blue=34, magenta=35, cyan=36, white=37, default=39)

   def __init__(self, stream):
       self.stream = stream

   @classmethod
   def supported(cls, stream=sys.stdout):
       """
       A class method that returns True if the current platform supports
       coloring terminal output using this method. Returns False otherwise.
       """
       if not stream.isatty():
           return False  # auto color only on TTYs
       try:
           import curses
       except ImportError:
           return False
       else:
           try:
               try:
                   return curses.tigetnum("colors") > 2
               except curses.error:
                   curses.setupterm()
                   return curses.tigetnum("colors") > 2
           except:
               raise
               # guess false in case of error
               return False

   def write(self, text, color):
       """
       Write the given text to the stream in the given color.

       @param text: Text to be written to the stream.

       @param color: A string label for a color. e.g. 'red', 'white'.
       """
       color = self._colors[color]
       self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))

class ColorHandler(logging.StreamHandler):

   def __init__(self, stream=sys.stdout):
       super(ColorHandler, self).__init__(_AnsiColorizer(stream))

   def emit(self, record):
       msg_colors = {
           logging.DEBUG: "green",
           logging.INFO: "default",
           logging.WARNING: "red",
           logging.ERROR: "red"
       }
       color = msg_colors.get(record.levelno, "blue")
       self.stream.write(record.msg + "\n", color)

class Logger(object):

   def __init__(self, name, log_dir, fold):
       self.pylogger = logging.getLogger(name)
       self.log_dir = log_dir
       self.fold = str(fold)

       self.pylogger.setLevel(logging.DEBUG)
       self.log_file = os.path.join(self.log_dir, name, "v" + self.fold + '.log')
       #self.log_file = os.path.join(log_dir, 'training.log')
       os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
       self.pylogger.addHandler(logging.FileHandler(self.log_file))
       self.pylogger.addHandler(ColorHandler())
       self.pylogger.propagate = False

   def __getattr__(self, attr):
       """delegate all undefined method requests to objects of
       this class in order pylogger, tboard (first find first serve).
       E.g., combinedlogger.add_scalars(...) should trigger self.tboard.add_scalars(...)
       """
       if attr in dir(self.pylogger):
           return getattr(self.pylogger, attr)
       print("logger attr not found")

   def set_logfile(self, version=None, log_file=None):
       if version is not None:
           self.fold = str(version)
       if log_file is None:
           self.log_file = os.path.join(self.log_dir, "v" + self.version, 'training.log')
       else:
           self.log_file = log_file
       os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
       for hdlr in self.pylogger.handlers:
           hdlr.close()
       self.pylogger.handlers = []
       self.pylogger.addHandler(logging.FileHandler(self.log_file))
       self.pylogger.addHandler(ColorHandler())

def get_logger(name, log_dir, version):
   """
   creates logger instance. writing out info to file, to terminal and to file.
   """
   logger = Logger(name, log_dir, version)
   print("Logging to {}".format(logger.log_file))
   return logger
