
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cufftdx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)

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

if(NOT TARGET cufftdx::cufftdx)
    set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)

    # commonDx dependency
    if(OFF)
        set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(TARGET mathdx::commondx)
        set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    endif()

    if(${cufftdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(cufftdx_VERSION "1.2.1")
        include("${CMAKE_CURRENT_LIST_DIR}/cufftdx-targets.cmake")

        # commonDx dependency
        if(OFF)
            set_and_check(cufftdx_commondx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/commondx/include")
            set_and_check(cufftdx_commondx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/commondx/include")
        elseif(TARGET mathdx::commondx)
            target_link_libraries(cufftdx::cufftdx INTERFACE mathdx::commondx)
            set_and_check(cufftdx_commondx_INCLUDE_DIR  "${mathdx_INCLUDE_DIR}")
            set_and_check(cufftdx_commondx_INCLUDE_DIRS "${mathdx_INCLUDE_DIR}")
        endif()

        set_and_check(cufftdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/cufftdx/include")
        set_and_check(cufftdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/cufftdx/include")
        set(cufftdx_LIBRARIES cufftdx::cufftdx)

        check_required_components(cufftdx)
        if(NOT cufftdx_FIND_QUIETLY)
            message(STATUS "Found cuFFTDx: (Location: ${cuFFTDx_LOCATION} Version: 1.2.1)")
        endif()
    endif()

    if(NOT ${cufftdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(cufftdx_FOUND FALSE)
        if(cufftdx_FIND_REQUIRED)
            message(FATAL_ERROR "cufftdx package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    endif()
endif()
