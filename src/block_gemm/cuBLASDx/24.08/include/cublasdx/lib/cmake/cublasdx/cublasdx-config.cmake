
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cublasdx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET cublasdx::cublasdx)
    set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)

    # commonDx dependency
    if(OFF)
        set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(TARGET mathdx::commondx)
        set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    endif()

    if((NOT DEFINED mathdx_DEPENDENCY_CUTLASS_RESOLVED) OR (NOT ${mathdx_DEPENDENCY_CUTLASS_RESOLVED}))
        # CUTLASS/CuTe dependency
        # Finds CUTLASS/CuTe and sets cublasdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
        #
        # CUTLASS root directory is found by checking following variables in this order:
        # 1. Root directory of NvidiaCutlass package
        # 2. mathdx_CUTLASS_ROOT
        # 3. ENV{mathdx_CUTLASS_ROOT}
        # 4. mathdx_CUTLASS_ROOT
        find_package(NvidiaCutlass QUIET)
        if(${NvidiaCutlass_FOUND})
            if(${NvidiaCutlass_VERSION} VERSION_LESS 3.5.1)
                message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is 3.5.1")
            endif()
            get_property(cublasdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            set_and_check(cublasdx_cutlass_INCLUDE_DIR "${cublasdx_NvidiaCutlass_include_dir}")
            if(NOT cublasdx_FIND_QUIETLY)
                message(STATUS "cublasdx: Found CUTLASS (NvidiaCutlass) dependency: ${cublasdx_NvidiaCutlass_include_dir}")
            endif()
            set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        elseif(DEFINED mathdx_CUTLASS_ROOT)
            get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE ${mathdx_CUTLASS_ROOT} ABSOLUTE)
            set_and_check(cublasdx_cutlass_INCLUDE_DIR  "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
            if(NOT cublasdx_FIND_QUIETLY)
                message(STATUS "cublasdx: Found CUTLASS dependency via mathdx_CUTLASS_ROOT: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
            endif()
            set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        elseif(DEFINED ENV{mathdx_CUTLASS_ROOT})
            get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE $ENV{mathdx_CUTLASS_ROOT} ABSOLUTE)
            set_and_check(cublasdx_cutlass_INCLUDE_DIR "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
            if(NOT cublasdx_FIND_QUIETLY)
                message(STATUS "cublasdx: Found CUTLASS dependency via ENV{mathdx_CUTLASS_ROOT}: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
            endif()
            set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        elseif(DEFINED mathdx_inc_CUTLASS_ROOT)
            get_filename_component(mathdx_inc_CUTLASS_ROOT_ABSOLUTE ${mathdx_inc_CUTLASS_ROOT} ABSOLUTE)
            set_and_check(cublasdx_cutlass_INCLUDE_DIR "${mathdx_inc_CUTLASS_ROOT_ABSOLUTE}/include")
            if(NOT cublasdx_FIND_QUIETLY)
                message(STATUS "cublasdx: Found CUTLASS shipped with MathDx package: ${mathdx_inc_CUTLASS_ROOT_ABSOLUTE}")
            endif()
            set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        endif()
    else()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR "${mathdx_cutlass_INCLUDE_DIR}")
    endif()

    if(${cublasdx_DEPENDENCY_CUTLASS_RESOLVED} AND ${cublasdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(cublasdx_VERSION "0.2.0")
        include("${CMAKE_CURRENT_LIST_DIR}/cublasdx-targets.cmake")

        # commonDx dependency
        if(OFF)
            set_and_check(cublasdx_commondx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/commondx/include")
            set_and_check(cublasdx_commondx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/commondx/include")
        elseif(TARGET mathdx::commondx)
            target_link_libraries(cublasdx::cublasdx INTERFACE mathdx::commondx)
            set_and_check(cublasdx_commondx_INCLUDE_DIR  "${mathdx_INCLUDE_DIR}")
            set_and_check(cublasdx_commondx_INCLUDE_DIRS "${mathdx_INCLUDE_DIR}")
        endif()

        # CUTLASS/CuTe dependency
        if(${NvidiaCutlass_FOUND})
            target_link_libraries(cublasdx::cublasdx INTERFACE nvidia::cutlass::cutlass)
        elseif(DEFINED mathdx_CUTLASS_ROOT)
            target_include_directories(cublasdx::cublasdx INTERFACE ${mathdx_cutlass_INCLUDE_DIR})
        elseif(DEFINED ENV{mathdx_CUTLASS_ROOT})
            target_include_directories(cublasdx::cublasdx INTERFACE ${mathdx_cutlass_INCLUDE_DIR})
        elseif(DEFINED mathdx_inc_CUTLASS_ROOT)
            target_include_directories(cublasdx::cublasdx INTERFACE ${mathdx_cutlass_INCLUDE_DIR})
        endif()

        set_and_check(cublasdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/mathdx/24.08/include/cublasdx/include")
        set_and_check(cublasdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/mathdx/24.08/include/cublasdx/include")
        set(cublasdx_LIBRARIES cublasdx::cublasdx)

        check_required_components(cublasdx)
        if(NOT cublasdx_FIND_QUIETLY)
            message(STATUS "Found cublasdx: (Location: ${cublasdx_LOCATION} Version: 0.2.0)")
        endif()
    endif()

    if(NOT ${cublasdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(cublasdx_FOUND FALSE)
        if(cublasdx_FIND_REQUIRED)
            message(FATAL_ERROR "cublasdx package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via mathdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    if(NOT ${cublasdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(cublasdx_FOUND FALSE)
        if(cublasdx_FIND_REQUIRED)
            message(FATAL_ERROR "cublasdx package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    endif()
endif()
