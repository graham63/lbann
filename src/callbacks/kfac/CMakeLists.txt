# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  kfac.cpp
  kfac_block.cpp
  kfac_block_bn.cpp
  kfac_block_fc_conv.cpp
  kfac_util.cpp
)

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    kfac_block_fc_conv.cu
    kfac_block_bn.cu
    kfac_util.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
