# =====================================
# The Boolean rule: how a box's fit groups combine
#
# One group: the fluid is its plane's fluid half-space. Several: they are connected through the
# patch edges between them, and every such edge must say the same thing --
#
#  * all convex -- the solid is the intersection of the groups' solid half-spaces (a transom
#    meeting the bottom, a chine);
#  * all concave -- the fluid is the intersection of their fluid half-spaces (a re-entrant corner);
#  * both -- a convex and a concave crease in one box (a chine flat, a transom corner ringed by
#    fillets) -- the mixed rule of `mixed.jl`, which labels the planes' arrangement from the mesh;
#  * groups with no patch edge between them (a feature thinner than the cell) are unsupported, and
#    the box is cut by one fallback plane and flagged.
#
# The chain rule accepts two patches with no shared edge when the box's other patches connect them:
# a narrow chine flat between the bottom and the side puts all three in one cell, with the bottom
# and the side never touching, and that corner is exactly what the method exists to get right.

"""
    _chain_rule(scr, s, pa, m, ca, cb, ng, np_total) -> (rule, flags)

The Boolean rule for slot `s`'s `ng` fit groups, from the labels between their member patches near
candidates `ca` and `cb` (0 for none), and `FLAG_UNSUPPORTED` when there is none.
"""
function _chain_rule(scr, s, pa, m, ca::Int, cb::Int, ng::Int, np_total::Int)
    ng <= 1 && return RULE_SINGLE, 0x00
    conv = false
    conc = false
    # Adjacency between groups, as a bit row per group.
    adj = ntuple(_ -> 0x00, Val(TRI_K_MAX))
    @inbounds for g in 1:ng, h in (g + 1):ng
        lab = 0x00
        mg = scr.gmem[g, s]
        mh = scr.gmem[h, s]
        for i in 1:TRI_K_MAX, j in 1:TRI_K_MAX
            ((mg >> (i - 1)) & 0x01 != 0 && (mh >> (j - 1)) & 0x01 != 0) || continue
            lab |= _pair_label(pa, m, ca, cb, scr.eid[i, s], scr.eid[j, s], np_total)
        end
        if lab & (PAIR_CONVEX | PAIR_CONCAVE) != 0x00
            adj = Base.setindex(adj, adj[g] | (UInt8(1) << (h - 1)), g)
            adj = Base.setindex(adj, adj[h] | (UInt8(1) << (g - 1)), h)
            conv |= lab & PAIR_CONVEX != 0x00
            conc |= lab & PAIR_CONCAVE != 0x00
        end
    end
    # Connected: grow from group 1 through the adjacency rows.
    reached = 0x01
    frontier = 0x01
    while frontier != 0x00
        g = trailing_zeros(frontier) + 1
        frontier &= ~(UInt8(1) << (g - 1))
        new = @inbounds adj[g] & ~reached
        reached |= new
        frontier |= new
    end
    all_groups = ng == 8 ? 0xff : (UInt8(1) << ng) - 0x01
    # Not connected: a feature thinner than the box, left to the fallback.
    reached == all_groups || return RULE_FALLBACK, FLAG_UNSUPPORTED
    # Connected with both senses: the planes' arrangement, labelled from the mesh (`mixed.jl`).
    (conv && conc) && return RULE_MIXED, 0x00
    return conc ? (RULE_CONCAVE, 0x00) : (RULE_CONVEX, 0x00)
end
