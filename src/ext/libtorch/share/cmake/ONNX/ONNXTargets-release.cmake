#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "onnx" for configuration "Release"
set_property(TARGET onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnx.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_onnx "${_IMPORT_PREFIX}/lib/onnx.lib" )

# Import target "onnx_proto" for configuration "Release"
set_property(TARGET onnx_proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnx_proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnx_proto.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS onnx_proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_onnx_proto "${_IMPORT_PREFIX}/lib/onnx_proto.lib" )

# Import target "onnxifi_dummy" for configuration "Release"
set_property(TARGET onnxifi_dummy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnxifi_dummy PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/onnxifi_dummy.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnxifi_dummy.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS onnxifi_dummy )
list(APPEND _IMPORT_CHECK_FILES_FOR_onnxifi_dummy "${_IMPORT_PREFIX}/lib/onnxifi_dummy.lib" "${_IMPORT_PREFIX}/lib/onnxifi_dummy.dll" )

# Import target "onnxifi_loader" for configuration "Release"
set_property(TARGET onnxifi_loader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnxifi_loader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnxifi_loader.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS onnxifi_loader )
list(APPEND _IMPORT_CHECK_FILES_FOR_onnxifi_loader "${_IMPORT_PREFIX}/lib/onnxifi_loader.lib" )

# Import target "onnxifi_wrapper" for configuration "Release"
set_property(TARGET onnxifi_wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnxifi_wrapper PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnxifi.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS onnxifi_wrapper )
list(APPEND _IMPORT_CHECK_FILES_FOR_onnxifi_wrapper "${_IMPORT_PREFIX}/lib/onnxifi.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
