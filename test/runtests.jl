using Pkg
Pkg.instantiate()

using ManifoldProjection, ConservativeDynamicalSystems, Random, Test

@testset "ManifoldProjection.jl" begin
    include("coordinate_projection.jl")
end
