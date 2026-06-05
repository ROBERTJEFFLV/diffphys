add_subdirectory(sac)
if(RL_TOOLS_BACKEND_ENABLE_CUDA AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    add_subdirectory(cuda)
endif()


add_executable(rl_environments_l2f_domain_randomization_sample dr_sample.cpp)
target_link_libraries(rl_environments_l2f_domain_randomization_sample PRIVATE RLtools::Core)


add_subdirectory(dr_sac)
