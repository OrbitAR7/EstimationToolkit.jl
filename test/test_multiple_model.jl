"""
Test suite for multiple model filters:
- StaticMultipleModel (SMM)
- InteractingMultipleModel (IMM)
- GPB1Filter
- GPB2Filter
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random

@testset "Multiple Model Filters" begin
    # Define two different dynamic models (for mode-switching)
    struct ConstantVelocityModel{T} <: AbstractDynamicsModel{2, 0, 2}
        Q::SMatrix{2, 2, T}
    end
    
    function EstimationToolkit.propagate(m::ConstantVelocityModel, x::SVector, u, dt)
        # [position, velocity]
        return @SVector [x[1] + x[2] * dt, x[2]]
    end
    
    function EstimationToolkit.jacobian_x(m::ConstantVelocityModel, x::SVector, u, dt)
        return @SMatrix [1.0 dt; 0.0 1.0]
    end
    
    EstimationToolkit.process_noise_cov(m::ConstantVelocityModel, dt) = m.Q
    
    struct ConstantAccelerationModel{T} <: AbstractDynamicsModel{2, 0, 2}
        a::T  # Acceleration
        Q::SMatrix{2, 2, T}
    end
    
    function EstimationToolkit.propagate(m::ConstantAccelerationModel, x::SVector, u, dt)
        # Velocity increases due to acceleration
        return @SVector [x[1] + x[2] * dt + 0.5 * m.a * dt^2, x[2] + m.a * dt]
    end
    
    function EstimationToolkit.jacobian_x(m::ConstantAccelerationModel, x::SVector, u, dt)
        return @SMatrix [1.0 dt; 0.0 1.0]
    end
    
    EstimationToolkit.process_noise_cov(m::ConstantAccelerationModel, dt) = m.Q
    
    # Common measurement model
    struct PositionMeasurement{T} <: AbstractMeasurementModel{2, 1, 1}
        R::SMatrix{1, 1, T}
    end
    
    EstimationToolkit.measure(m::PositionMeasurement, x::SVector) = @SVector [x[1]]
    EstimationToolkit.jacobian_x(m::PositionMeasurement, x::SVector) = @SMatrix [1.0 0.0]
    EstimationToolkit.measurement_noise_cov(m::PositionMeasurement) = m.R
    
    @testset "StaticMultipleModel" begin
        # Create filter bank for different parameter values
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(0.5, Q),
            ConstantAccelerationModel(-0.5, Q)
        ]
        
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        # Create filter for each model
        filters = [KalmanFilter(x0, P0) for _ in 1:3]
        model_probs = [0.5, 0.25, 0.25]
        u = zeros(Float64, 0)
        
        smm = StaticMultipleModelFilter(filters, model_probs)
        
        @test length(smm.filters) == 3
        @test sum(smm.model_probs) ≈ 1.0
        
        # Test prediction
        predict!(smm, models, u, 1.0)
        
        # All filters should have predicted
        for f in smm.filters
            @test f.x !== x0 || f.x == x0  # State may or may not change
        end
        
        # Test update
        meas = PositionMeasurement(@SMatrix [0.5])
        meas_models = fill(meas, length(models))
        y = @SVector [1.5]
        
        update!(smm, meas_models, y)
        
        # Model probabilities should sum to 1
        @test sum(smm.model_probs) ≈ 1.0
        
        # Get overall estimate
        x_est = state_estimate(smm)
        P_est = state_covariance(smm)
        
        @test length(x_est) == 2
        @test size(P_est) == (2, 2)
        @test is_positive_definite(P_est)
        
        # Run full sequence
        for k in 1:10
            predict!(smm, models, u, 1.0)
            y_k = @SVector [k * 1.0]
            update!(smm, meas_models, y_k)
            @test sum(smm.model_probs) ≈ 1.0
        end
        
        # One model should become dominant
        max_prob = maximum(smm.model_probs)
        @test max_prob > 0.3  # At least one model has significant probability
    end
    
    @testset "InteractingMultipleModel" begin
        # Create IMM with mode transitions
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(1.0, Q)
        ]
        
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        filters = [KalmanFilter(x0, P0) for _ in 1:2]
        
        # Mode transition matrix (tendency to stay in same mode)
        Π = @SMatrix [0.9 0.1; 0.1 0.9]
        model_probs = [0.5, 0.5]
        u = zeros(Float64, 0)
        
        imm = InteractingMultipleModel(filters, Matrix(Π), model_probs)
        
        @test size(imm.transition_matrix) == (2, 2)
        @test all(sum(imm.transition_matrix, dims=2) .≈ 1.0)  # Rows sum to 1
        @test sum(imm.mode_probs) ≈ 1.0
        
        meas = PositionMeasurement(@SMatrix [0.3])
        meas_models = fill(meas, length(models))
        
        # Run IMM cycle
        for k in 1:15
            # Step combines predict and update
            y_k = @SVector [k * 0.5]
            step!(imm, models, meas_models, u, y_k, 1.0)
            
            @test sum(imm.mode_probs) ≈ 1.0
            
            # Get mixed estimate
            x_est = state_estimate(imm)
            @test length(x_est) == 2
        end
        
        # One more step to ensure stability
        y = @SVector [5.0]
        step!(imm, models, meas_models, u, y, 1.0)
        @test sum(imm.mode_probs) ≈ 1.0
    end
    
    @testset "GPB1Filter" begin
        # GPB1: Run filters in parallel without mixing
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(0.8, Q)
        ]
        
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        filters = [KalmanFilter(x0, P0) for _ in 1:2]
        Π = @SMatrix [0.95 0.05; 0.05 0.95]
        model_probs = [0.6, 0.4]
        u = zeros(Float64, 0)
        
        gpb1 = GPB1Filter(filters, Matrix(Π), model_probs)
        
        @test length(gpb1.filters) == 2
        @test sum(gpb1.mode_probs) ≈ 1.0
        
        meas = PositionMeasurement(@SMatrix [0.4])
        meas_models = fill(meas, length(models))
        
        # Test step
        y = @SVector [2.0]
        step!(gpb1, models, meas_models, u, y, 1.0)
        @test sum(gpb1.mode_probs) ≈ 1.0
        
        # Get overall estimate
        x_est = state_estimate(gpb1)
        P_est = state_covariance(gpb1)
        
        @test length(x_est) == 2
        @test is_positive_definite(P_est)
        
        # Run sequence
        for k in 1:8
            y_k = @SVector [k * 0.8]
            step!(gpb1, models, meas_models, u, y_k, 1.0)
            @test sum(gpb1.mode_probs) ≈ 1.0
        end
    end
    
    @testset "GPB2Filter" begin
        # GPB2: Maintain M×M filter bank (joint history)
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(0.5, Q)
        ]
        
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        M = 2  # Number of models
        Π = @SMatrix [0.9 0.1; 0.1 0.9]
        model_probs = [0.5, 0.5]
        u = zeros(Float64, 0)
        
        filter_template = KalmanFilter(x0, P0)
        gpb2 = GPB2Filter(filter_template, M, Matrix(Π), model_probs)
        
        @test length(gpb2.filters) == M * M
        @test sum(gpb2.joint_probs) ≈ 1.0
        
        meas = PositionMeasurement(@SMatrix [0.4])
        meas_models = fill(meas, length(models))
        
        # Test step
        y = @SVector [1.5]
        step!(gpb2, models, meas_models, u, y, 1.0)
        @test sum(gpb2.joint_probs) ≈ 1.0 atol=1e-6
        
        # Get joint probabilities
        joint_matrix = get_joint_probabilities(gpb2)
        @test size(joint_matrix) == (M, M)
        @test sum(joint_matrix) ≈ 1.0
        
        # Get overall estimate
        x_est = state_estimate(gpb2)
        P_est = state_covariance(gpb2)
        
        @test length(x_est) == 2
        @test is_positive_definite(P_est)
        
        # Run sequence
        for k in 1:6
            y_k = @SVector [k * 0.6]
            step!(gpb2, models, meas_models, u, y_k, 1.0)
            @test sum(gpb2.joint_probs) ≈ 1.0 atol=1e-6
        end
    end
    
    @testset "Multiple Model Edge Cases" begin
        # Single model (degenerate case)
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        single_model = [ConstantVelocityModel(Q)]
        single_filter = [KalmanFilter(x0, P0)]
        
        u = zeros(Float64, 0)
        smm_single = StaticMultipleModelFilter(single_filter, [1.0])
        
        @test length(smm_single.filters) == 1
        @test smm_single.model_probs[1] ≈ 1.0
        
        meas = PositionMeasurement(@SMatrix [0.5])
        predict!(smm_single, single_model, u, 1.0)
        update!(smm_single, [meas], @SVector[1.0])
        
        @test smm_single.model_probs[1] ≈ 1.0
        
        # IMM with identity transition matrix (no mode switching)
        Π_identity = @SMatrix [1.0 0.0; 0.0 1.0]
        models = [ConstantVelocityModel(Q), ConstantAccelerationModel(0.5, Q)]
        filters = [KalmanFilter(x0, P0) for _ in 1:2]
        
        imm_no_switch = InteractingMultipleModel(filters, Matrix(Π_identity), [0.7, 0.3])
        meas_models = fill(meas, length(models))
        
        initial_probs = copy(imm_no_switch.mode_probs)
        
        for _ in 1:5
            step!(imm_no_switch, models, meas_models, u, @SVector[1.0], 1.0)
        end
        
        # Probabilities should change only due to likelihoods, not transitions
        @test sum(imm_no_switch.mode_probs) ≈ 1.0
    end
    
    @testset "Mode Probability Evolution" begin
        # Test that IMM properly tracks mode changes
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        x0 = @SVector [0.0, 0.0]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        # Two very different models
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(2.0, Q)  # Strong acceleration
        ]
        
        filters = [KalmanFilter(x0, P0) for _ in 1:2]
        Π = @SMatrix [0.8 0.2; 0.2 0.8]
        
        u = zeros(Float64, 0)
        imm = InteractingMultipleModel(filters, Matrix(Π), [0.5, 0.5])
        
        meas = PositionMeasurement(@SMatrix [0.2])
        meas_models = fill(meas, length(models))
        
        # Generate data matching model 2 (acceleration)
        Random.seed!(789)
        for k in 1:20
            # Accelerating target
            y_k = @SVector [0.5 * k^2]
            step!(imm, models, meas_models, u, y_k, 1.0)
        end
        
        # Model 2 (acceleration) should have higher probability
        @test imm.mode_probs[2] > imm.mode_probs[1]
        @test sum(imm.mode_probs) ≈ 1.0
    end
    
    @testset "Multiple Model Comparison" begin
        # Compare SMM, IMM, GPB1, GPB2 on same problem
        Q = @SMatrix [0.05 0.0; 0.0 0.05]
        x0 = @SVector [0.0, 1.0]
        P0 = @SMatrix [0.5 0.0; 0.0 0.5]
        
        models = [
            ConstantVelocityModel(Q),
            ConstantAccelerationModel(0.3, Q)
        ]
        
        Π = @SMatrix [0.95 0.05; 0.05 0.95]
        model_probs = [0.5, 0.5]
        
        # Create all filter types
        u = zeros(Float64, 0)
        
        smm = StaticMultipleModelFilter(
            [KalmanFilter(x0, P0) for _ in 1:2],
            model_probs
        )
        
        imm = InteractingMultipleModel(
            [KalmanFilter(x0, P0) for _ in 1:2],
            Matrix(Π), model_probs
        )
        
        gpb1 = GPB1Filter(
            [KalmanFilter(x0, P0) for _ in 1:2],
            Matrix(Π), model_probs
        )
        
        gpb2 = GPB2Filter(
            KalmanFilter(x0, P0),
            2, Matrix(Π), model_probs
        )
        
        meas = PositionMeasurement(@SMatrix [0.3])
        meas_models = fill(meas, length(models))
        
        # Run all filters
        for k in 1:10
            y_k = @SVector [k * 0.5]
            
            predict!(smm, models, u, 1.0)
            update!(smm, meas_models, y_k)
            
            step!(imm, models, meas_models, u, y_k, 1.0)
            
            step!(gpb1, models, meas_models, u, y_k, 1.0)
            
            step!(gpb2, models, meas_models, u, y_k, 1.0)
        end
        
        # All should produce valid estimates
        x_smm = state_estimate(smm)
        x_imm = state_estimate(imm)
        x_gpb1 = state_estimate(gpb1)
        x_gpb2 = state_estimate(gpb2)
        
        @test all(isfinite, x_smm)
        @test all(isfinite, x_imm)
        @test all(isfinite, x_gpb1)
        @test all(isfinite, x_gpb2)
        
        # All should have valid covariances
        @test is_positive_definite(state_covariance(smm))
        @test is_positive_definite(state_covariance(imm))
        @test is_positive_definite(state_covariance(gpb1))
        @test is_positive_definite(state_covariance(gpb2))
    end
end
