#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace rl_tools::foundation_policy::equivalent_dynamics{

    template <typename T>
    struct LatentTarget{
        T thrust_to_acceleration_gain = 0;
        T roll_pitch_torque_to_angular_acceleration_gain = 0;
        T yaw_torque_to_angular_acceleration_gain = 0;
        T motor_rise_time_constant = 0;
        T motor_fall_time_constant = 0;
        T thrust_curve_shape = 0;
        T torque_curve_shape = 0;
        T residual_force_bias = 0;
        T residual_torque_bias = 0;
    };

    template <typename T>
    struct NormalizedLatentTarget{
        T thrust_to_acceleration_gain = 0;
        T roll_pitch_torque_to_angular_acceleration_gain = 0;
        T yaw_torque_to_angular_acceleration_gain = 0;
        T motor_rise_time_constant = 0;
        T motor_fall_time_constant = 0;
        T thrust_curve_shape = 0;
        T torque_curve_shape = 0;
        T residual_force_bias = 0;
        T residual_torque_bias = 0;
    };

    template <typename TI>
    struct DynamicsBins{
        TI size_mass = 0;
        TI thrust_to_weight = 0;
        TI torque_to_inertia = 0;
        TI motor_delay = 0;
        TI curve_shape = 0;
    };

    template <typename T>
    T finite_or_zero(T value){
        return value == value && value > (T)-1e20 && value < (T)1e20 ? value : (T)0;
    }

    template <typename T>
    T safe_abs(T value){
        return std::abs(finite_or_zero(value));
    }

    template <typename T>
    T safe_div(T numerator, T denominator, T fallback = 0){
        return safe_abs(denominator) > (T)1e-12 ? numerator / denominator : fallback;
    }

    template <typename T>
    T clamp01(T value){
        return std::max((T)0, std::min((T)1, value));
    }

    template <typename T, typename TI>
    TI bin_index(T value, T min_value, T max_value, TI num_bins){
        if(num_bins <= 1 || max_value <= min_value){
            return 0;
        }
        const T normalized = clamp01((value - min_value) / (max_value - min_value));
        TI index = (TI)std::floor(normalized * (T)num_bins);
        return std::min(num_bins - 1, std::max((TI)0, index));
    }

    template <typename T, typename TI>
    void restrict_range_to_bin(T& min_value, T& max_value, TI bin, TI num_bins){
        if(num_bins <= 1 || max_value <= min_value){
            return;
        }
        const T original_min = min_value;
        const T width = (max_value - min_value) / (T)num_bins;
        const TI safe_bin = std::min(num_bins - 1, std::max((TI)0, bin));
        min_value = original_min + width * (T)safe_bin;
        max_value = original_min + width * (T)(safe_bin + 1);
    }

    template <typename DOMAIN, typename T, typename TI>
    void restrict_domain_to_bins(DOMAIN& domain, const DynamicsBins<TI>& bins, TI num_bins){
        restrict_range_to_bin<T, TI>(domain.mass_min, domain.mass_max, bins.size_mass, num_bins);
        restrict_range_to_bin<T, TI>(domain.thrust_to_weight_min, domain.thrust_to_weight_max, bins.thrust_to_weight, num_bins);
        restrict_range_to_bin<T, TI>(domain.torque_to_inertia_min, domain.torque_to_inertia_max, bins.torque_to_inertia, num_bins);
        restrict_range_to_bin<T, TI>(domain.rotor_time_constant_rising_min, domain.rotor_time_constant_rising_max, bins.motor_delay, num_bins);
        restrict_range_to_bin<T, TI>(domain.rotor_time_constant_falling_min, domain.rotor_time_constant_falling_max, bins.motor_delay, num_bins);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T gravity_norm(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        T g2 = 0;
        for(typename PARAMETERS::TI dim_i = 0; dim_i < 3; dim_i++){
            g2 += parameters.dynamics.gravity[dim_i] * parameters.dynamics.gravity[dim_i];
        }
        return std::sqrt(std::max((T)0, g2));
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T rotor_force_at(const PARAMETERS& parameters, typename PARAMETERS::T rpm, typename PARAMETERS::TI rotor_i){
        const auto& c = parameters.dynamics.rotor_thrust_coefficients[rotor_i];
        return c[0] + c[1] * rpm + c[2] * rpm * rpm;
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T max_total_thrust_body_z_abs(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        T thrust_z = 0;
        const T max_rpm = parameters.dynamics.action_limit.max;
        for(typename PARAMETERS::TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
            thrust_z += parameters.dynamics.rotor_thrust_directions[rotor_i][2] * rotor_force_at<PARAMETERS>(parameters, max_rpm, rotor_i);
        }
        return std::abs(thrust_z);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T thrust_to_weight_ratio(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        return safe_div(max_total_thrust_body_z_abs(parameters), parameters.dynamics.mass * gravity_norm(parameters));
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T thrust_to_acceleration_gain(const PARAMETERS& parameters){
        return safe_div(max_total_thrust_body_z_abs(parameters), parameters.dynamics.mass);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T roll_pitch_torque_to_inertia_gain(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        const T g = std::max((T)1e-6, gravity_norm(parameters));
        const T max_thrust_per_rotor = thrust_to_weight_ratio(parameters) * parameters.dynamics.mass * g / (T)PARAMETERS::N;
        const T rotor_distance_x = std::abs(parameters.dynamics.rotor_positions[0][0]);
        const T rotor_distance_y = std::abs(parameters.dynamics.rotor_positions[0][1]);
        const T effective_arm = std::sqrt(rotor_distance_x * rotor_distance_x + rotor_distance_y * rotor_distance_y);
        const T max_roll_pitch_torque = effective_arm * max_thrust_per_rotor;
        const T mean_roll_pitch_inertia = ((safe_abs(parameters.dynamics.J[0][0]) + safe_abs(parameters.dynamics.J[1][1])) / (T)2);
        return safe_div(max_roll_pitch_torque, mean_roll_pitch_inertia);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T yaw_torque_to_inertia_gain(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        const T max_rpm = parameters.dynamics.action_limit.max;
        T yaw_torque = 0;
        for(typename PARAMETERS::TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
            const T force = rotor_force_at<PARAMETERS>(parameters, max_rpm, rotor_i);
            yaw_torque += std::abs(parameters.dynamics.rotor_torque_directions[rotor_i][2] * parameters.dynamics.rotor_torque_constants[rotor_i] * force);
        }
        return safe_div(yaw_torque, parameters.dynamics.J[2][2]);
    }

    template <typename PARAMETERS>
    void motor_time_constants(const PARAMETERS& parameters, typename PARAMETERS::T& rise_mean, typename PARAMETERS::T& fall_mean, typename PARAMETERS::T& mean, typename PARAMETERS::T& min_value, typename PARAMETERS::T& max_value){
        using T = typename PARAMETERS::T;
        rise_mean = 0;
        fall_mean = 0;
        min_value = std::numeric_limits<T>::infinity();
        max_value = -std::numeric_limits<T>::infinity();
        for(typename PARAMETERS::TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
            const T rising = parameters.dynamics.rotor_time_constants_rising[rotor_i];
            const T falling = parameters.dynamics.rotor_time_constants_falling[rotor_i];
            rise_mean += rising;
            fall_mean += falling;
            min_value = std::min(min_value, std::min(rising, falling));
            max_value = std::max(max_value, std::max(rising, falling));
        }
        rise_mean /= (T)PARAMETERS::N;
        fall_mean /= (T)PARAMETERS::N;
        mean = (rise_mean + fall_mean) / (T)2;
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T motor_time_constant_mean(const PARAMETERS& parameters){
        typename PARAMETERS::T rise, fall, mean, min_value, max_value;
        motor_time_constants<PARAMETERS>(parameters, rise, fall, mean, min_value, max_value);
        return mean;
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T thrust_curve_shape(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        const T max_rpm = parameters.dynamics.action_limit.max;
        T shape = 0;
        for(typename PARAMETERS::TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
            const auto& c = parameters.dynamics.rotor_thrust_coefficients[rotor_i];
            const T c0 = safe_abs(c[0]);
            const T c1 = safe_abs(c[1] * max_rpm);
            const T c2 = safe_abs(c[2] * max_rpm * max_rpm);
            shape += safe_div(c2, c0 + c1 + c2);
        }
        return shape / (T)PARAMETERS::N;
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T torque_curve_shape(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        T torque_gain_sum = 0;
        for(typename PARAMETERS::TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
            torque_gain_sum += safe_abs(parameters.dynamics.rotor_torque_constants[rotor_i]);
        }
        const T mean_torque_gain = torque_gain_sum / (T)PARAMETERS::N;
        return mean_torque_gain * thrust_curve_shape<PARAMETERS>(parameters);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T residual_force_bias(const PARAMETERS& parameters){
        return safe_abs(parameters.disturbances.random_force.mean) + safe_abs(parameters.disturbances.random_force.std);
    }

    template <typename PARAMETERS>
    typename PARAMETERS::T residual_torque_bias(const PARAMETERS& parameters){
        return safe_abs(parameters.disturbances.random_torque.mean) + safe_abs(parameters.disturbances.random_torque.std);
    }

    template <typename PARAMETERS>
    LatentTarget<typename PARAMETERS::T> target_from_parameters(const PARAMETERS& parameters){
        using T = typename PARAMETERS::T;
        T rise, fall, mean, min_value, max_value;
        motor_time_constants<PARAMETERS>(parameters, rise, fall, mean, min_value, max_value);
        LatentTarget<T> target;
        target.thrust_to_acceleration_gain = thrust_to_acceleration_gain<PARAMETERS>(parameters);
        target.roll_pitch_torque_to_angular_acceleration_gain = roll_pitch_torque_to_inertia_gain<PARAMETERS>(parameters);
        target.yaw_torque_to_angular_acceleration_gain = yaw_torque_to_inertia_gain<PARAMETERS>(parameters);
        target.motor_rise_time_constant = rise;
        target.motor_fall_time_constant = fall;
        target.thrust_curve_shape = thrust_curve_shape<PARAMETERS>(parameters);
        target.torque_curve_shape = torque_curve_shape<PARAMETERS>(parameters);
        target.residual_force_bias = residual_force_bias<PARAMETERS>(parameters);
        target.residual_torque_bias = residual_torque_bias<PARAMETERS>(parameters);
        return target;
    }

    template <typename PARAMETERS>
    NormalizedLatentTarget<typename PARAMETERS::T> normalized_target_from_parameters(
        const PARAMETERS& parameters,
        const PARAMETERS& nominal
    ){
        using T = typename PARAMETERS::T;
        const auto target = target_from_parameters<PARAMETERS>(parameters);
        const auto nominal_target = target_from_parameters<PARAMETERS>(nominal);
        NormalizedLatentTarget<T> normalized;
        normalized.thrust_to_acceleration_gain = safe_div(target.thrust_to_acceleration_gain, nominal_target.thrust_to_acceleration_gain);
        normalized.roll_pitch_torque_to_angular_acceleration_gain = safe_div(target.roll_pitch_torque_to_angular_acceleration_gain, nominal_target.roll_pitch_torque_to_angular_acceleration_gain);
        normalized.yaw_torque_to_angular_acceleration_gain = safe_div(target.yaw_torque_to_angular_acceleration_gain, nominal_target.yaw_torque_to_angular_acceleration_gain);
        normalized.motor_rise_time_constant = safe_div(target.motor_rise_time_constant, nominal_target.motor_rise_time_constant);
        normalized.motor_fall_time_constant = safe_div(target.motor_fall_time_constant, nominal_target.motor_fall_time_constant);
        normalized.thrust_curve_shape = target.thrust_curve_shape;
        normalized.torque_curve_shape = safe_div(target.torque_curve_shape, nominal_target.torque_curve_shape);
        normalized.residual_force_bias = target.residual_force_bias;
        normalized.residual_torque_bias = target.residual_torque_bias;
        return normalized;
    }

    template <typename PARAMETERS, typename TI>
    DynamicsBins<TI> bins_from_parameters(const PARAMETERS& parameters, TI num_bins){
        using T = typename PARAMETERS::T;
        const auto& domain = parameters.domain_randomization;
        const T tau_mean = motor_time_constant_mean<PARAMETERS>(parameters);
        const T tau_min = (domain.rotor_time_constant_rising_min + domain.rotor_time_constant_falling_min) / (T)2;
        const T tau_max = (domain.rotor_time_constant_rising_max + domain.rotor_time_constant_falling_max) / (T)2;
        DynamicsBins<TI> bins;
        bins.size_mass = bin_index<T, TI>(parameters.dynamics.mass, domain.mass_min, domain.mass_max, num_bins);
        bins.thrust_to_weight = bin_index<T, TI>(thrust_to_weight_ratio<PARAMETERS>(parameters), domain.thrust_to_weight_min, domain.thrust_to_weight_max, num_bins);
        bins.torque_to_inertia = bin_index<T, TI>(roll_pitch_torque_to_inertia_gain<PARAMETERS>(parameters), domain.torque_to_inertia_min, domain.torque_to_inertia_max, num_bins);
        bins.motor_delay = bin_index<T, TI>(tau_mean, tau_min, tau_max, num_bins);
        bins.curve_shape = bin_index<T, TI>(thrust_curve_shape<PARAMETERS>(parameters), (T)0, (T)1, num_bins);
        return bins;
    }
}
