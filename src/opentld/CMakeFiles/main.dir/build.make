# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/martina/projects/OpenTLD-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/martina/projects/OpenTLD-master

# Include any dependencies generated for this target.
include src/opentld/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include src/opentld/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include src/opentld/CMakeFiles/main.dir/flags.make

src/opentld/CMakeFiles/main.dir/main/Config.cpp.o: src/opentld/CMakeFiles/main.dir/flags.make
src/opentld/CMakeFiles/main.dir/main/Config.cpp.o: src/opentld/main/Config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/opentld/CMakeFiles/main.dir/main/Config.cpp.o"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main/Config.cpp.o -c /home/martina/projects/OpenTLD-master/src/opentld/main/Config.cpp

src/opentld/CMakeFiles/main.dir/main/Config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main/Config.cpp.i"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martina/projects/OpenTLD-master/src/opentld/main/Config.cpp > CMakeFiles/main.dir/main/Config.cpp.i

src/opentld/CMakeFiles/main.dir/main/Config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main/Config.cpp.s"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martina/projects/OpenTLD-master/src/opentld/main/Config.cpp -o CMakeFiles/main.dir/main/Config.cpp.s

src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.requires:

.PHONY : src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.requires

src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.provides: src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.requires
	$(MAKE) -f src/opentld/CMakeFiles/main.dir/build.make src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.provides.build
.PHONY : src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.provides

src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.provides.build: src/opentld/CMakeFiles/main.dir/main/Config.cpp.o


src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o: src/opentld/CMakeFiles/main.dir/flags.make
src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o: src/opentld/main/Gui.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main/Gui.cpp.o -c /home/martina/projects/OpenTLD-master/src/opentld/main/Gui.cpp

src/opentld/CMakeFiles/main.dir/main/Gui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main/Gui.cpp.i"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martina/projects/OpenTLD-master/src/opentld/main/Gui.cpp > CMakeFiles/main.dir/main/Gui.cpp.i

src/opentld/CMakeFiles/main.dir/main/Gui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main/Gui.cpp.s"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martina/projects/OpenTLD-master/src/opentld/main/Gui.cpp -o CMakeFiles/main.dir/main/Gui.cpp.s

src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.requires:

.PHONY : src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.requires

src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.provides: src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.requires
	$(MAKE) -f src/opentld/CMakeFiles/main.dir/build.make src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.provides.build
.PHONY : src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.provides

src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.provides.build: src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o


src/opentld/CMakeFiles/main.dir/main/Main.cpp.o: src/opentld/CMakeFiles/main.dir/flags.make
src/opentld/CMakeFiles/main.dir/main/Main.cpp.o: src/opentld/main/Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/opentld/CMakeFiles/main.dir/main/Main.cpp.o"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main/Main.cpp.o -c /home/martina/projects/OpenTLD-master/src/opentld/main/Main.cpp

src/opentld/CMakeFiles/main.dir/main/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main/Main.cpp.i"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martina/projects/OpenTLD-master/src/opentld/main/Main.cpp > CMakeFiles/main.dir/main/Main.cpp.i

src/opentld/CMakeFiles/main.dir/main/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main/Main.cpp.s"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martina/projects/OpenTLD-master/src/opentld/main/Main.cpp -o CMakeFiles/main.dir/main/Main.cpp.s

src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.requires:

.PHONY : src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.requires

src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.provides: src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.requires
	$(MAKE) -f src/opentld/CMakeFiles/main.dir/build.make src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.provides.build
.PHONY : src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.provides

src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.provides.build: src/opentld/CMakeFiles/main.dir/main/Main.cpp.o


src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o: src/opentld/CMakeFiles/main.dir/flags.make
src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o: src/opentld/main/Settings.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main/Settings.cpp.o -c /home/martina/projects/OpenTLD-master/src/opentld/main/Settings.cpp

src/opentld/CMakeFiles/main.dir/main/Settings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main/Settings.cpp.i"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martina/projects/OpenTLD-master/src/opentld/main/Settings.cpp > CMakeFiles/main.dir/main/Settings.cpp.i

src/opentld/CMakeFiles/main.dir/main/Settings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main/Settings.cpp.s"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martina/projects/OpenTLD-master/src/opentld/main/Settings.cpp -o CMakeFiles/main.dir/main/Settings.cpp.s

src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.requires:

.PHONY : src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.requires

src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.provides: src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.requires
	$(MAKE) -f src/opentld/CMakeFiles/main.dir/build.make src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.provides.build
.PHONY : src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.provides

src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.provides.build: src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o


src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o: src/opentld/CMakeFiles/main.dir/flags.make
src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o: src/opentld/main/Trajectory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main/Trajectory.cpp.o -c /home/martina/projects/OpenTLD-master/src/opentld/main/Trajectory.cpp

src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main/Trajectory.cpp.i"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martina/projects/OpenTLD-master/src/opentld/main/Trajectory.cpp > CMakeFiles/main.dir/main/Trajectory.cpp.i

src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main/Trajectory.cpp.s"
	cd /home/martina/projects/OpenTLD-master/src/opentld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martina/projects/OpenTLD-master/src/opentld/main/Trajectory.cpp -o CMakeFiles/main.dir/main/Trajectory.cpp.s

src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.requires:

.PHONY : src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.requires

src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.provides: src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.requires
	$(MAKE) -f src/opentld/CMakeFiles/main.dir/build.make src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.provides.build
.PHONY : src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.provides

src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.provides.build: src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main/Config.cpp.o" \
"CMakeFiles/main.dir/main/Gui.cpp.o" \
"CMakeFiles/main.dir/main/Main.cpp.o" \
"CMakeFiles/main.dir/main/Settings.cpp.o" \
"CMakeFiles/main.dir/main/Trajectory.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

lib/libmain.a: src/opentld/CMakeFiles/main.dir/main/Config.cpp.o
lib/libmain.a: src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o
lib/libmain.a: src/opentld/CMakeFiles/main.dir/main/Main.cpp.o
lib/libmain.a: src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o
lib/libmain.a: src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o
lib/libmain.a: src/opentld/CMakeFiles/main.dir/build.make
lib/libmain.a: src/opentld/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/martina/projects/OpenTLD-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library ../../lib/libmain.a"
	cd /home/martina/projects/OpenTLD-master/src/opentld && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean_target.cmake
	cd /home/martina/projects/OpenTLD-master/src/opentld && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/opentld/CMakeFiles/main.dir/build: lib/libmain.a

.PHONY : src/opentld/CMakeFiles/main.dir/build

src/opentld/CMakeFiles/main.dir/requires: src/opentld/CMakeFiles/main.dir/main/Config.cpp.o.requires
src/opentld/CMakeFiles/main.dir/requires: src/opentld/CMakeFiles/main.dir/main/Gui.cpp.o.requires
src/opentld/CMakeFiles/main.dir/requires: src/opentld/CMakeFiles/main.dir/main/Main.cpp.o.requires
src/opentld/CMakeFiles/main.dir/requires: src/opentld/CMakeFiles/main.dir/main/Settings.cpp.o.requires
src/opentld/CMakeFiles/main.dir/requires: src/opentld/CMakeFiles/main.dir/main/Trajectory.cpp.o.requires

.PHONY : src/opentld/CMakeFiles/main.dir/requires

src/opentld/CMakeFiles/main.dir/clean:
	cd /home/martina/projects/OpenTLD-master/src/opentld && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : src/opentld/CMakeFiles/main.dir/clean

src/opentld/CMakeFiles/main.dir/depend:
	cd /home/martina/projects/OpenTLD-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/martina/projects/OpenTLD-master /home/martina/projects/OpenTLD-master/src/opentld /home/martina/projects/OpenTLD-master /home/martina/projects/OpenTLD-master/src/opentld /home/martina/projects/OpenTLD-master/src/opentld/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/opentld/CMakeFiles/main.dir/depend

