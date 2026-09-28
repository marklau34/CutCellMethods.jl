struct CutCellTriangulatedProblem{M<:Mesh, G<:CartesianGrid}
    mesh::M
    grid::G
end

# Store the result as a array of structs
struct CutCellTriangulatedResult
end

function CommonSolve.solve(prob::CutCellTriangulatedProblem)
end

function CommonSolve.solve!(system::CutCellTriangulatedResult)
end