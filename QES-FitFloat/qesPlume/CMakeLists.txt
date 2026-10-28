# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

#IF ($CACHE{HAS_CUDA_SUPPORT})
#  CUDA_ADD_EXECUTABLE( qesPlume
#          handlePlumeArgs.cpp
#          qesPlumeMain.cpp
#  )
#ELSE ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE( qesPlume
          handlePlumeArgs.cpp
          qesPlumeMain.cpp
  )
#ENDIF ($CACHE{HAS_CUDA_SUPPORT})

target_link_libraries(qesPlume PRIVATE qesplumecore)

IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qesPlume PRIVATE qeswindsgpu)
ENDIF()
target_link_libraries(qesPlume PRIVATE qeswindscore)

target_link_libraries(qesPlume PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesPlume PRIVATE qesOptix)
ENDIF()

link_external_libraries(qesPlume)

# FitFloat study: benchmark driver for the GPU particle model (src/plume/CUDA), see qesPlumeGPU.cpp
IF ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE(qesPlumeGPU qesPlumeGPU.cpp)
  target_include_directories(qesPlumeGPU PRIVATE ${PROJECT_SOURCE_DIR}/src/plume)   # Partition.h includes "Particle.h"
  SET_TARGET_PROPERTIES(qesPlumeGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(qesPlumeGPU PRIVATE qesplumegpu)
  target_link_libraries(qesPlumeGPU PRIVATE qesplumecore)
  target_link_libraries(qesPlumeGPU PRIVATE qeswindsgpu)
  target_link_libraries(qesPlumeGPU PRIVATE qeswindscore)
  target_link_libraries(qesPlumeGPU PRIVATE qesutilgpu)
  target_link_libraries(qesPlumeGPU PRIVATE qesutil)
  target_link_libraries(qesPlumeGPU PRIVATE CUDA::cudart CUDA::curand)
  IF ($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(qesPlumeGPU PRIVATE qesOptix)
  ENDIF()
  link_external_libraries(qesPlumeGPU)
ENDIF()
