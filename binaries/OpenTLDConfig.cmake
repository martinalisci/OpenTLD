find_package(OpenCV)

set(bin_dir "/home/martina/Scrivania/OpenTLD-master(1)/OpenTLD-master/binaries")
set(src_dir "/home/martina/Scrivania/OpenTLD-master(1)/OpenTLD-master")

set(OPENTLD_INCLUDE_DIRS
    ${src_dir}/src/3rdparty/cvblobs
    ${src_dir}/src/3rdparty/libconfig
    ${src_dir}/src/libopentld/tld
    ${src_dir}/src/libopentld/mftracker
    ${src_dir}/src/libopentld/imacq
    ${src_dir}/src/opentld/main)
INCLUDE_DIRECTORIES(${OPENTLD_INCLUDE_DIRS})

SET(OPENTLD_LIB_DIRS ${LIBRARY_OUTPUT_PATH})
LINK_DIRECTORIES(${OPENTLD_LIB_DIRS})

set(OPENTLD_LIBS cvblobs libconfig libopentld)



