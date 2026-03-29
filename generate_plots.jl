#=
Gera os gráficos para o artigo PQ sobre dedicação/MILP/shadow prices.
Roda após dedication.jl (que define todas as variáveis).
Usa o cenário ×20 (liabilities_large) para os gráficos do artigo.
Salva PNGs na pasta artigos/ do pq_dedication_article.
=#

using Plots, Printf

output_dir = raw"C:\Users\andre\pq_projects\pq_dedication_article\artigos"

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1: Curva implícita vs curva de mercado
# ─────────────────────────────────────────────────────────────────────────────
# Nota: os shadow prices da relaxação LP são os mesmos independente da escala
# dos passivos (a base ótima do LP não muda). Usamos os do cenário base.

rates_market = [(1.0 / d_market[t])^(1.0 / t) - 1.0 for t in 1:T]
rates_implicit = [(1.0 / abs(shadow_prices[t]))^(1.0 / t) - 1.0 for t in 1:T]

p1 = plot(1:T, rates_market .* 100,
    label = "Curva de mercado (bootstrapping)",
    marker = :circle, markersize = 6, linewidth = 2,
    color = "#1a5276",
    xlabel = "Ano", ylabel = "Taxa spot (%)",
    title = "Estrutura a Termo: Implícita vs Mercado",
    legend = :topright, legendfontsize = 9,
    size = (800, 450), dpi = 200,
    grid = true, gridalpha = 0.3,
    titlefontsize = 13, guidefontsize = 11,
    tickfontsize = 10, margin = 5Plots.mm
)
plot!(p1, 1:T, rates_implicit .* 100,
    label = "Curva implícita (shadow prices)",
    marker = :diamond, markersize = 6, linewidth = 2,
    color = "#FF6719"
)
# Anotar o spread do ano 1
annotate!(p1, 1.3, rates_implicit[1] * 100 - 0.5,
    text("+463 bps", 9, :left, "#FF6719"))

savefig(p1, joinpath(output_dir, "grafico_curvas.png"))
println("Salvo: grafico_curvas.png")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2: Análise paramétrica — LP vs MILP (cenário ×20)
# ─────────────────────────────────────────────────────────────────────────────

deltas_plot = collect(-6000:1000:10000)
sp5 = shadow_prices[5]
lp_base_val = objective_value(model_lp_lg)

lp_costs = Float64[]
milp_costs = Float64[]
sp_pred = Float64[]

for δ in deltas_plot
    liab_mod = copy(liabilities_large)
    liab_mod[5] += δ

    m_lp, _, _ = solve_dedication_lp(bonds, cf, liab_mod, reinvest_rate)
    m_milp, _, _ = solve_dedication_milp(bonds, cf, liab_mod, reinvest_rate)

    push!(lp_costs, objective_value(m_lp))
    push!(milp_costs, objective_value(m_milp))
    push!(sp_pred, lp_base_val + sp5 * δ)
end

p2 = plot(deltas_plot, milp_costs,
    label = "Custo MILP (lotes inteiros)",
    linewidth = 2.5,
    color = "#c0392b",
    xlabel = "Δ passivo Ano 5 (USD)", ylabel = "Custo da carteira (USD)",
    title = "Sensibilidade: Custo LP vs MILP",
    legend = :topleft, legendfontsize = 9,
    size = (800, 450), dpi = 200,
    grid = true, gridalpha = 0.3,
    titlefontsize = 13, guidefontsize = 11,
    tickfontsize = 10, margin = 5Plots.mm
)
plot!(p2, deltas_plot, lp_costs,
    label = "Custo LP (contínuo)",
    linewidth = 2, color = "#1a5276"
)
plot!(p2, deltas_plot, sp_pred,
    label = "Previsão shadow price",
    linewidth = 1.5, linestyle = :dash, color = "#FF6719"
)

savefig(p2, joinpath(output_dir, "grafico_parametrica.png"))
println("Salvo: grafico_parametrica.png")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 3: Perfil de surplus — MILP (cenário ×20)
# ─────────────────────────────────────────────────────────────────────────────

recebido = [sum(cf[j, t] * round(Int, value(x_milp_lg[j])) for j in 1:J) for t in 1:T]
surplus_reinv = [t > 1 ? (1 + reinvest_rate) * value(s_milp_lg[t-1]) : 0.0 for t in 1:T]
surplus_final = [value(s_milp_lg[t]) for t in 1:T]

p3 = plot(size = (800, 450), dpi = 200,
    title = "Perfil de Caixa da Carteira MILP",
    xlabel = "Ano", ylabel = "USD",
    legend = :topright, legendfontsize = 9,
    grid = true, gridalpha = 0.3,
    titlefontsize = 13, guidefontsize = 11,
    tickfontsize = 10, margin = 5Plots.mm
)
bar!(p3, 1:T, recebido,
    label = "Recebido (bonds)", color = "#2e86c1", alpha = 0.8, bar_width = 0.35,
    bar_position = -0.2)
bar!(p3, 1:T, surplus_reinv,
    label = "Surplus reinvestido", color = "#FF6719", alpha = 0.8, bar_width = 0.35,
    bar_position = 0.2)
plot!(p3, 1:T, liabilities_large,
    label = "Passivo", marker = :square, markersize = 5,
    linewidth = 2.5, color = "#1a1a2e", linestyle = :solid)
plot!(p3, 1:T, surplus_final,
    label = "Surplus final", marker = :circle, markersize = 5,
    linewidth = 2, color = "#27ae60", linestyle = :dash)

savefig(p3, joinpath(output_dir, "grafico_surplus.png"))
println("Salvo: grafico_surplus.png")

println("\nTodos os gráficos gerados com sucesso!")
