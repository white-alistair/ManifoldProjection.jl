@testset "coordinate_projection.jl" begin
    @testset "One Invariant" begin
        Random.seed!(1)
        perturbation_scale = 1e-3
        abstol = 1e-9

        system = DoublePendulum{Float64}(rand(4)...)
        u0 = rand(4)
        t0 = rand()
        initial_invariants = invariants(u0, system, t0)

        perturbed_state = u0 .+ perturbation_scale * randn(4)
        perturbed_invariants = invariants(perturbed_state, system, t0)

        residual_function = (u, t) -> invariants(u, system, t0) - initial_invariants
        residual_jacobian = (u, t) -> invariants_jacobian(u, system, t)
        coordinate_projection = CoordinateProjection(residual_function, residual_jacobian, abstol)

        projected_state = coordinate_projection(perturbed_state, t0)
        projected_invariants = invariants(projected_state, system, t0)

        @test !≈(initial_invariants, perturbed_invariants, atol = abstol)
        @test ≈(projected_invariants, initial_invariants, atol = abstol)
    end
end
