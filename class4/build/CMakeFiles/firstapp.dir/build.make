# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /root/projects/class4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/projects/class4/build

# Include any dependencies generated for this target.
include CMakeFiles/firstapp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/firstapp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/firstapp.dir/flags.make

CMakeFiles/firstapp.dir/main.cu.o: CMakeFiles/firstapp.dir/flags.make
CMakeFiles/firstapp.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/projects/class4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/firstapp.dir/main.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/projects/class4/main.cu -o CMakeFiles/firstapp.dir/main.cu.o

CMakeFiles/firstapp.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/firstapp.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/firstapp.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/firstapp.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target firstapp
firstapp_OBJECTS = \
"CMakeFiles/firstapp.dir/main.cu.o"

# External object files for target firstapp
firstapp_EXTERNAL_OBJECTS =

firstapp: CMakeFiles/firstapp.dir/main.cu.o
firstapp: CMakeFiles/firstapp.dir/build.make
firstapp: CMakeFiles/firstapp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/projects/class4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable firstapp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/firstapp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/firstapp.dir/build: firstapp

.PHONY : CMakeFiles/firstapp.dir/build

CMakeFiles/firstapp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/firstapp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/firstapp.dir/clean

CMakeFiles/firstapp.dir/depend:
	cd /root/projects/class4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/projects/class4 /root/projects/class4 /root/projects/class4/build /root/projects/class4/build /root/projects/class4/build/CMakeFiles/firstapp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/firstapp.dir/depend
