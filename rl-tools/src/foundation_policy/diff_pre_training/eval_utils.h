#pragma once

#include "config.h"
#include "loop.h"

#include <rl_tools/rl/environments/l2f/diff_euler_model.h>
#include <rl_tools/rl/environments/l2f/diff_euler_rollout.h>
#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/containers/tensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    template <typename T>
    T norm3(const T v[3]){
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    template <typename T>
    struct ScalarTerms{
        T total = 0;
        T position = 0;
        T velocity = 0;
        T attitude = 0;
        T angular_velocity = 0;
        T action_magnitude = 0;
        T action_smoothness = 0;
        T saturation = 0;
        T terminal = 0;
        T terminal_position = 0;
        T terminal_velocity = 0;
        T terminal_attitude = 0;
        T terminal_angular_velocity = 0;
        T clf = 0;
        T window_clf = 0;
        T outward_velocity = 0;
        T attitude_control = 0;
        T velocity_barrier = 0;
        T angular_velocity_barrier = 0;
        T attitude_barrier = 0;
    };

    template <typename T>
    struct RolloutMetrics{
        T initial_position_norm = 0;
        T final_position_norm = 0;
        T initial_velocity_norm = 0;
        T final_velocity_norm = 0;
        T initial_angular_velocity_norm = 0;
        T final_angular_velocity_norm = 0;
        T action_norm = 0;
        T preclamp_action_norm = 0;
        T postclamp_action_norm = 0;
        T max_action_norm = 0;
        T action_clamp_rate = 0;
        T final_state_norm = 0;
        bool invalid = false;
    };

    template <typename T>
    struct EvalMetrics{
        T success_rate = 0;
        T mean_total_loss = 0;
        T mean_final_position_norm = 0;
        T mean_final_velocity_norm = 0;
        T mean_final_angular_velocity_norm = 0;
        T median_final_position_norm = 0;
        T median_final_velocity_norm = 0;
        T median_final_angular_velocity_norm = 0;
        T p90_final_position_norm = 0;
        T p90_final_velocity_norm = 0;
        T near_success_rate_p = 0;
        T near_success_rate_pv = 0;
        T mean_action_norm = 0;
        T invalid_or_nan_rate = 0;
    };

    template <typename T>
    bool finite_value(T value){
        return value == value && value > (T)-1e20 && value < (T)1e20;
    }

    template <typename T>
    bool finite_scalar_terms(const ScalarTerms<T>& terms){
        return finite_value(terms.total) && finite_value(terms.position) && finite_value(terms.velocity) &&
               finite_value(terms.attitude) && finite_value(terms.angular_velocity) &&
               finite_value(terms.action_magnitude) && finite_value(terms.action_smoothness) && finite_value(terms.saturation) &&
               finite_value(terms.terminal) && finite_value(terms.terminal_position) && finite_value(terms.terminal_velocity) &&
               finite_value(terms.terminal_attitude) && finite_value(terms.terminal_angular_velocity) &&
               finite_value(terms.clf) && finite_value(terms.outward_velocity) && finite_value(terms.attitude_control);
    }

    template <typename T>
    T clip_value(T value, T limit){
        if(value > limit) return limit;
        if(value < -limit) return -limit;
        return value;
    }

    template <typename T>
    T clamp01(T value){
        if(value < (T)0) return (T)0;
        if(value > (T)1) return (T)1;
        return value;
    }

    template <typename T, typename TI>
    void add_metrics(RolloutMetrics<T>& accumulator, const RolloutMetrics<T>& value){
        accumulator.initial_position_norm += value.initial_position_norm;
        accumulator.final_position_norm += value.final_position_norm;
        accumulator.initial_velocity_norm += value.initial_velocity_norm;
        accumulator.final_velocity_norm += value.final_velocity_norm;
        accumulator.initial_angular_velocity_norm += value.initial_angular_velocity_norm;
        accumulator.final_angular_velocity_norm += value.final_angular_velocity_norm;
        accumulator.action_norm += value.action_norm;
        accumulator.preclamp_action_norm += value.preclamp_action_norm;
        accumulator.postclamp_action_norm += value.postclamp_action_norm;
        accumulator.max_action_norm = std::max(accumulator.max_action_norm, value.max_action_norm);
        accumulator.action_clamp_rate += value.action_clamp_rate;
        accumulator.final_state_norm += value.final_state_norm;
        accumulator.invalid = accumulator.invalid || value.invalid;
    }

    namespace l2f_diff = rl_tools::rl::environments::l2f::diff;

    template <typename T, typename TI>
    T euler_position_error_norm(const l2f_diff::EulerState<T, TI>& state, const l2f_diff::TrackingReference<T>& ref){
        T error[3];
        for(TI i = 0; i < 3; i++){
            error[i] = state.p[i] - ref.p[i];
        }
        return norm3(error);
    }

    template <typename T, typename TI>
    T euler_velocity_error_norm(const l2f_diff::EulerState<T, TI>& state, const l2f_diff::TrackingReference<T>& ref){
        T error[3];
        for(TI i = 0; i < 3; i++){
            error[i] = state.v[i] - ref.v[i];
        }
        return norm3(error);
    }

    template <typename STATE, typename T, typename TI>
    T l2f_position_error_norm(const STATE& state, const l2f_diff::TrackingReference<T>& ref){
        T error[3];
        for(TI i = 0; i < 3; i++){
            error[i] = state.position[i] - ref.p[i];
        }
        return norm3(error);
    }

    template <typename STATE, typename T, typename TI>
    T l2f_velocity_error_norm(const STATE& state, const l2f_diff::TrackingReference<T>& ref){
        T error[3];
        for(TI i = 0; i < 3; i++){
            error[i] = state.linear_velocity[i] - ref.v[i];
        }
        return norm3(error);
    }

    template <typename T, typename TI>
    ScalarTerms<T> euler_state_loss_terms(const l2f_diff::EulerState<T, TI>& state, const l2f_diff::TrackingReference<T>& ref, const l2f_diff::EulerLossWeights<T>& weights){
        ScalarTerms<T> terms;
        for(TI i = 0; i < 3; i++){
            const T e_p = state.p[i] - ref.p[i];
            const T e_v = state.v[i] - ref.v[i];
            terms.position += weights.position * e_p * e_p;
            terms.velocity += weights.velocity * e_v * e_v;
            terms.angular_velocity += weights.angular_velocity * state.omega[i] * state.omega[i];
        }
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                const T target = i == j ? (T)1 : (T)0;
                const T error = state.R[i][j] - target;
                terms.attitude += weights.attitude * error * error;
            }
        }
        terms.total = terms.position + terms.velocity + terms.attitude + terms.angular_velocity;
        return terms;
    }

    template <typename T, typename TI>
    ScalarTerms<T> euler_state_loss_terms(const l2f_diff::EulerState<T, TI>& state, const l2f_diff::EulerLossWeights<T>& weights){
        const auto ref = l2f_diff::zero_tracking_reference<T>();
        return euler_state_loss_terms<T, TI>(state, ref, weights);
    }

    template <typename STATE, typename T, typename TI>
    ScalarTerms<T> l2f_state_loss_terms(const STATE& state, const l2f_diff::TrackingReference<T>& ref, const LossWeights<T>& weights){
        ScalarTerms<T> terms;
        const T orientation_sign = state.orientation[0] >= (T)0 ? (T)1 : (T)-1;
        for(TI i = 0; i < 3; i++){
            const T e_p = state.position[i] - ref.p[i];
            const T e_v = state.linear_velocity[i] - ref.v[i];
            const T e_R = (T)2 * orientation_sign * state.orientation[i + 1];
            terms.position += weights.position * e_p * e_p;
            terms.velocity += weights.velocity * e_v * e_v;
            terms.attitude += weights.attitude * e_R * e_R;
            terms.angular_velocity += weights.angular_velocity * state.angular_velocity[i] * state.angular_velocity[i];
        }
        terms.total = terms.position + terms.velocity + terms.attitude + terms.angular_velocity;
        return terms;
    }

    template <typename STATE, typename T, typename TI>
    ScalarTerms<T> l2f_state_loss_terms(const STATE& state, const LossWeights<T>& weights){
        const auto ref = l2f_diff::zero_tracking_reference<T>();
        return l2f_state_loss_terms<STATE, T, TI>(state, ref, weights);
    }
}
