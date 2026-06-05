add_executable(foundation_policy_diff_physics_check diff_pre_training/physics_check.cpp)
target_link_libraries(foundation_policy_diff_physics_check PRIVATE RLtools::RLtools)
target_compile_options(foundation_policy_diff_physics_check PRIVATE -ftemplate-backtrace-limit=0)

add_executable(foundation_policy_diff_pre_training diff_pre_training/main.cpp)
target_link_libraries(foundation_policy_diff_pre_training PRIVATE RLtools::RLtools)
target_compile_options(foundation_policy_diff_pre_training PRIVATE -ftemplate-backtrace-limit=0)

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_executable(
        foundation_policy_diff_pre_training_cuda
        diff_pre_training/cuda_main.cu
        diff_pre_training/gpu_rollout.cu
    )
    target_include_directories(foundation_policy_diff_pre_training_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(TARGET CUDA::cudart)
        target_link_libraries(foundation_policy_diff_pre_training_cuda PRIVATE CUDA::cudart)
    endif()
    target_compile_options(
        foundation_policy_diff_pre_training_cuda
        PRIVATE
        -ftemplate-backtrace-limit=0
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    )
    set_target_properties(
        foundation_policy_diff_pre_training_cuda
        PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

add_executable(foundation_policy_actor_equivalence_check post_training/actor_equivalence_check.cpp)
target_link_libraries(foundation_policy_actor_equivalence_check PRIVATE RLtools::RLtools)
target_compile_options(foundation_policy_actor_equivalence_check PRIVATE -ftemplate-backtrace-limit=0)

if(RL_TOOLS_ENABLE_HDF5)
    add_executable(foundation_policy_pre_training pre_training/main.cpp)
    target_link_libraries(foundation_policy_pre_training PRIVATE RLtools::RLtools)
    target_compile_options(foundation_policy_pre_training PRIVATE -ftemplate-backtrace-limit=0)

    add_executable(foundation_policy_post_training post_training/main.cpp)
    target_link_libraries(foundation_policy_post_training PRIVATE RLtools::RLtools)
    target_compile_options(foundation_policy_post_training PRIVATE -ftemplate-backtrace-limit=0)

    add_executable(foundation_policy_post_training_evaluate post_training/evaluate_student.cpp)
    target_link_libraries(foundation_policy_post_training_evaluate PRIVATE RLtools::RLtools)
    target_compile_options(foundation_policy_post_training_evaluate PRIVATE -ftemplate-backtrace-limit=0)
endif()


add_executable(foundation_policy_pre_training_sample_dynamics_parameters pre_training/sample_dynamics_parameters.cpp)
target_link_libraries(foundation_policy_pre_training_sample_dynamics_parameters PRIVATE RLtools::RLtools)
target_compile_options(foundation_policy_pre_training_sample_dynamics_parameters PRIVATE -ftemplate-backtrace-limit=0)
