#pragma once

#include <fstream>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training{

    inline void write_training_csv_header(std::ofstream& log_file){
        log_file << "seed,step,mode_fixed_or_sampled,diff_model,physics_gradient_enabled,reset_hidden_each_step,"
                 << "current_horizon,current_state_difficulty,current_dynamics_difficulty,"
                 << "sampled_dynamics_level,dynamics_sampled_flag,dynamics_sample_rejected_count,dynamics_rejection_rate,"
                 << "init_actor_path,init_actor_loaded_flag,"
                 << "h128_prioritized_curriculum_enabled,h128_schedule_name,curriculum_phase_name,phase_index,steps_in_current_phase,"
                 << "horizon_changed_flag,state_curriculum_changed_flag,steps_since_horizon_change,steps_since_state_change,"
                 << "effective_terminal_velocity_weight,effective_terminal_angular_velocity_weight,terminal_ramp_multiplier,"
                 << "optimizer_reset_flag,"
                 << "rejected_dynamics_count,valid_rollout_count,invalid_rollout_count,"
                 << "mass,thrust_to_weight_ratio,torque_to_inertia_ratio,motor_tau_mean,motor_tau_min,motor_tau_max,"
                 << "sampled_tau_rise,sampled_tau_fall,sampled_curve_shape,sampled_parameters_inside_allowed_ranges,"
                 << "inertia_trace,thrust_scale,torque_scale,"
                 << "balanced_dynamics_sampling,dynamics_num_groups,"
                 << "dynamics_group_count_min,dynamics_group_count_max,dynamics_group_weight_sum_min,dynamics_group_weight_sum_max,dynamics_batch_balanced,"
                 << "dynamics_size_mass_bin,dynamics_thrust_to_weight_bin,"
                 << "dynamics_torque_to_inertia_bin,dynamics_motor_delay_bin,dynamics_curve_shape_bin,"
                 << "equivalent_dynamics_diag_thrust_to_acceleration_gain,"
                 << "equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain,"
                 << "equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain,"
                 << "equivalent_dynamics_diag_motor_rise_time_constant,equivalent_dynamics_diag_motor_fall_time_constant,"
                 << "equivalent_dynamics_diag_thrust_curve_shape,equivalent_dynamics_diag_torque_curve_shape,"
                 << "equivalent_dynamics_diag_residual_force_bias,equivalent_dynamics_diag_residual_torque_bias,"
                 << "total_loss_mean,position_loss_mean,velocity_loss_mean,attitude_loss_mean,"
                 << "angular_velocity_loss_mean,action_magnitude_loss_mean,action_smoothness_loss_mean,"
                 << "saturation_loss_mean,terminal_loss_mean,terminal_position_loss_mean,"
                 << "terminal_velocity_loss_mean,terminal_attitude_loss_mean,terminal_angular_velocity_loss_mean,"
                 << "transition_consistency_loss,"
                 << "actor_critic_actor_loss,actor_critic_critic_loss,"
                 << "single_step_state_finite,single_step_action_finite,single_step_reward_finite,single_step_done_finite,"
                 << "diff_rollout_loss_weight,"
                 << "rdac_encoder_grad_norm_before_clip,rdac_gru_grad_norm_before_clip,"
                 << "rdac_actor_head_grad_norm_before_clip,rdac_critic_head_grad_norm_before_clip,"
                 << "rdac_encoder_grad_norm_after_clip,rdac_gru_grad_norm_after_clip,"
                 << "rdac_actor_head_grad_norm_after_clip,rdac_critic_head_grad_norm_after_clip,"
                 << "hidden_dynamics_separation_ratio,hidden_dynamics_between_var,hidden_dynamics_within_var,hidden_dynamics_separable,"
                 << "action_grad_norm_before_clip,action_grad_norm_after_clip,action_grad_scale,action_grad_clipped_flag,"
                 << "actor_grad_norm_before_clip,actor_grad_norm_after_clip,actor_grad_scale,actor_grad_clipped_flag,"
                 << "actor_grad_nan_or_inf_flag,"
                 << "actor_step_skipped,actor_step_applied,num_skipped_steps,num_applied_steps,skip_reason,"
                 << "mean_initial_position_norm,mean_final_position_norm,mean_initial_velocity_norm,"
                 << "mean_final_velocity_norm,mean_initial_angular_velocity_norm,mean_final_angular_velocity_norm,"
                 << "mean_action_norm,mean_preclamp_action_norm,mean_postclamp_action_norm,"
                 << "max_action_norm,action_clamp_rate,nan_or_inf_flag,"
                 << "loss_total,loss_position,loss_velocity,loss_angular_velocity,"
                 << "loss_terminal_position,loss_terminal_velocity,loss_terminal_angular_velocity,"
                 << "loss_action_magnitude,loss_action_smoothness,loss_action_saturation,"
                 << "loss_stabilization,loss_action_regularization,"
                 << "actor_grad_norm_pre_clip,actor_grad_norm_post_clip,actor_grad_max_abs,actor_update_norm,"
                 << "actor_parameter_norm,actor_param_norm,actor_param_max_abs,actor_update_to_param_norm_ratio,adam_m_norm,adam_v_norm,"
                 << "action_grad_norm_pre_clip,action_grad_norm_post_clip,action_grad_clip_scale,action_grad_clip_active_flag,"
                 << "action_mean,action_std,action_min,action_max,action_abs_mean,action_saturation_ratio,"
                 << "action_delta_mean,action_delta_max,training_success_rate,"
                 << "rpm_mean,rpm_min,rpm_max,thrust_mean,torque_norm_mean,"
                 << "gru_hidden_norm_mean,gru_hidden_norm_max,gru_hidden_abs_mean,gru_hidden_abs_max,gru_hidden_nan_or_inf_flag,"
                 << "diagnostic_unavailable_fields\n";
    }

    inline void write_eval_csv_header(std::ofstream& eval_log){
        eval_log << "eval_model,mode_fixed_or_sampled,eval_dynamics_mode,eval_sampled_dynamics_level,eval_episodes,eval_horizon,"
                 << "success_rate,near_success_rate_p,near_success_rate_pv,"
                 << "mean_total_loss,mean_final_position_norm,mean_final_velocity_norm,"
                 << "mean_final_angular_velocity_norm,median_final_position_norm,"
                 << "median_final_velocity_norm,median_final_angular_velocity_norm,"
                 << "p90_final_position_norm,p90_final_velocity_norm,"
                 << "mean_action_norm,invalid_or_nan_rate\n";
    }
}
