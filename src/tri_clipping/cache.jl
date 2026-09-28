# Store the result as a array of structs
struct TriClippingCutCellCache <: AbstractCutCellCache
end

function allocate_cache(grid::CartesianGrid, ::TriClippingCutCell)
    return TriClippingCutCellCache()
end

function update_cache!(cache::TriClippingCutCellCache, mesh::M, grid::CartesianGrid) where {M}
end