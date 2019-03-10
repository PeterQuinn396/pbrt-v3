#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "c10_cuda" for configuration "Release"
set_property(TARGET c10_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/c10_cuda.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/c10_cuda.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS c10_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_c10_cuda "${_IMPORT_PREFIX}/lib/c10_cuda.lib" "${_IMPORT_PREFIX}/lib/c10_cuda.dll" )

# Import target "c10" for configuration "Release"
set_property(TARGET c10 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10 PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/c10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/c10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS c10 )
list(APPEND _IMPORT_CHECK_FILES_FOR_c10 "${_IMPORT_PREFIX}/lib/c10.lib" "${_IMPORT_PREFIX}/lib/c10.dll" )

# Import target "caffe2" for configuration "Release"
set_property(TARGET caffe2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe2 PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffe2.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/caffe2.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe2 "${_IMPORT_PREFIX}/lib/caffe2.lib" "${_IMPORT_PREFIX}/lib/caffe2.dll" )

# Import target "caffe2_gpu" for configuration "Release"
set_property(TARGET caffe2_gpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe2_gpu PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffe2_gpu.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/caffe2_gpu.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe2_gpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe2_gpu "${_IMPORT_PREFIX}/lib/caffe2_gpu.lib" "${_IMPORT_PREFIX}/lib/caffe2_gpu.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)