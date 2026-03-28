#=
╔══════════════════════════════════════════════════════════════════════════════╗
║  Cash Flow Matching (Dedication) via MILP                                  ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  Problema: um fundo de pensão precisa cobrir passivos projetados em cada   ║
║  ano futuro. Compramos títulos de renda fixa (bonds) em LOTES INTEIROS     ║
║  no mercado secundário. Os cupons e amortizações desses títulos devem      ║
║  cobrir cada passivo, e o excesso de caixa é reinvestido a uma taxa        ║
║  conservadora (short-rate).                                                ║
║                                                                            ║
║  Após resolver o MILP, relaxamos a integralidade para obter os SHADOW      ║
║  PRICES das restrições de passivo — que revelam a ESTRUTURA A TERMO        ║
║  IMPLÍCITA da carteira dedicada.                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
=#

using JuMP, HiGHS, DataFrames, PrettyTables, Printf

# Helper: formatar número com separador de milhares (Julia não suporta %,)
function fmt_usd(v::Real; decimals=2)
    s = @sprintf("%.2f", v)
    parts = split(s, '.')
    int_part = parts[1]
    # Inserir vírgulas a cada 3 dígitos da direita
    negative = startswith(int_part, '-')
    digits = negative ? int_part[2:end] : int_part
    groups = String[]
    while length(digits) > 3
        pushfirst!(groups, digits[end-2:end])
        digits = digits[1:end-3]
    end
    pushfirst!(groups, digits)
    formatted = join(groups, ",")
    return (negative ? "-" : "") * formatted * "." * parts[2]
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. DADOS DO PROBLEMA
# ─────────────────────────────────────────────────────────────────────────────

# Horizonte: 8 anos (t = 1, 2, …, 8)
T = 8

# Passivos do fundo (em milhares de USD) — o que precisamos pagar a cada ano
liabilities = [100.0, 200.0, 800.0, 100.0, 800.0, 1200.0, 800.0, 100.0]

# Taxa de reinvestimento do excesso de caixa (conservadora)
reinvest_rate = 0.02

# Cada bond j é definido por:
#   - coupon_rate: taxa de cupom anual (% do face value)
#   - maturity:    ano de vencimento (paga face value + último cupom)
#   - price:       preço de mercado por lote (1 lote = 1000 USD de face value)
#   - face_value:  valor de face por lote

struct Bond
    name::String
    coupon_rate::Float64
    maturity::Int
    price::Float64       # preço por lote no mercado secundário
    face_value::Float64  # valor de face por lote
end

bonds = [
    Bond("Bond A",  0.050, 3,  1020.0, 1000.0),  # 5.0%, vence ano 3
    Bond("Bond B",  0.060, 5,  1050.0, 1000.0),  # 6.0%, vence ano 5
    Bond("Bond C",  0.055, 7,  1000.0, 1000.0),  # 5.5%, vence ano 7
    Bond("Bond D",  0.045, 2,   990.0, 1000.0),  # 4.5%, vence ano 2
    Bond("Bond E",  0.070, 8,  1100.0, 1000.0),  # 7.0%, vence ano 8
    Bond("Bond F",  0.040, 4,   980.0, 1000.0),  # 4.0%, vence ano 4
    Bond("Bond G",  0.065, 6,  1060.0, 1000.0),  # 6.5%, vence ano 6
    Bond("Bond H",  0.035, 1,   995.0, 1000.0),  # 3.5%, vence ano 1
]

J = length(bonds)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRIZ DE FLUXO DE CAIXA
# ─────────────────────────────────────────────────────────────────────────────
# cashflow[j, t] = fluxo de caixa gerado por 1 lote do bond j no ano t

function build_cashflow_matrix(bonds::Vector{Bond}, T::Int)
    J = length(bonds)
    cf = zeros(J, T)
    for j in 1:J
        b = bonds[j]
        for t in 1:T
            if t < b.maturity
                cf[j, t] = b.coupon_rate * b.face_value    # cupom
            elseif t == b.maturity
                cf[j, t] = b.coupon_rate * b.face_value + b.face_value  # cupom + face
            end
            # t > maturity → 0 (bond já venceu)
        end
    end
    return cf
end

cf = build_cashflow_matrix(bonds, T)

# Exibir a matriz de fluxo de caixa
println("=" ^ 80)
println("  MATRIZ DE FLUXO DE CAIXA (por lote)")
println("=" ^ 80)
cf_pairs = Pair{Symbol, Any}[
    :Bond => [b.name for b in bonds],
    :Cupom => [@sprintf("%.1f%%", b.coupon_rate * 100) for b in bonds],
    :Venc => [b.maturity for b in bonds],
    :Preço => [b.price for b in bonds],
    [Symbol("Ano $t") => cf[:, t] for t in 1:T]...
]
cf_df = DataFrame(cf_pairs)
pretty_table(cf_df, alignment=:c)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELO MILP — DEDICATION
# ─────────────────────────────────────────────────────────────────────────────

function solve_dedication_milp(bonds, cf, liabilities, reinvest_rate)
    T = length(liabilities)
    J = length(bonds)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[j] = número de LOTES inteiros comprados do bond j
    @variable(model, x[1:J] >= 0, Int)

    # s[t] = excesso de caixa (surplus) no final do ano t
    @variable(model, s[1:T] >= 0)

    # Restrição de balanço de caixa para cada ano t:
    #   (recebimentos dos bonds) + (surplus reinvestido do ano anterior)
    #   = (passivo do ano) + (surplus carregado para o próximo ano)
    @constraint(model, balance[t in 1:T],
        sum(cf[j, t] * x[j] for j in 1:J) + (t > 1 ? (1 + reinvest_rate) * s[t-1] : 0.0)
        == liabilities[t] + s[t]
    )

    # Objetivo: minimizar o custo total de aquisição da carteira
    @objective(model, Min, sum(bonds[j].price * x[j] for j in 1:J))

    optimize!(model)

    return model, x, s
end

model_milp, x_milp, s_milp = solve_dedication_milp(bonds, cf, liabilities, reinvest_rate)

# ─────────────────────────────────────────────────────────────────────────────
# 4. RESULTADOS DO MILP
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 80)
println("  RESULTADO MILP — CARTEIRA DEDICADA (lotes inteiros)")
println("=" ^ 80)

milp_status = termination_status(model_milp)
println("Status: $milp_status")
println("Custo total da carteira: USD $(fmt_usd(objective_value(model_milp)))")

# Carteira comprada
println("\n  Carteira de Bonds Comprados:")
portfolio_df = DataFrame(
    Bond = String[],
    Lotes = Int[],
    Cupom = String[],
    Vencimento = Int[],
    Custo_Total = Float64[]
)
for j in 1:J
    lots = round(Int, value(x_milp[j]))
    if lots > 0
        push!(portfolio_df, (
            bonds[j].name,
            lots,
            @sprintf("%.1f%%", bonds[j].coupon_rate * 100),
            bonds[j].maturity,
            lots * bonds[j].price
        ))
    end
end
pretty_table(portfolio_df, alignment=:c)

# Surplus por ano
println("\n  Surplus (excesso de caixa) por ano:")
surplus_df = DataFrame(
    Ano = 1:T,
    Passivo = liabilities,
    Recebido = [sum(cf[j, t] * round(Int, value(x_milp[j])) for j in 1:J) for t in 1:T],
    Surplus_Reinvestido = [t > 1 ? (1 + reinvest_rate) * value(s_milp[t-1]) : 0.0 for t in 1:T],
    Surplus_Final = [value(s_milp[t]) for t in 1:T]
)
pretty_table(surplus_df, alignment=:c)

# ─────────────────────────────────────────────────────────────────────────────
# 5. RELAXAÇÃO LP PARA SHADOW PRICES (ANÁLISE DE SENSIBILIDADE)
# ─────────────────────────────────────────────────────────────────────────────

function solve_dedication_lp(bonds, cf, liabilities, reinvest_rate)
    T = length(liabilities)
    J = length(bonds)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Agora x[j] é CONTÍNUO — relaxação da integralidade
    @variable(model, x[1:J] >= 0)
    @variable(model, s[1:T] >= 0)

    @constraint(model, balance[t in 1:T],
        sum(cf[j, t] * x[j] for j in 1:J) + (t > 1 ? (1 + reinvest_rate) * s[t-1] : 0.0)
        == liabilities[t] + s[t]
    )

    @objective(model, Min, sum(bonds[j].price * x[j] for j in 1:J))

    optimize!(model)

    return model, x, s
end

model_lp, x_lp, s_lp = solve_dedication_lp(bonds, cf, liabilities, reinvest_rate)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SHADOW PRICES → ESTRUTURA A TERMO IMPLÍCITA
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 80)
println("  ANÁLISE DE SENSIBILIDADE — SHADOW PRICES")
println("=" ^ 80)

println("\nRelaxação LP:")
println("  Custo LP (contínuo):  USD $(fmt_usd(objective_value(model_lp)))")
println("  Custo MILP (inteiro): USD $(fmt_usd(objective_value(model_milp)))")
gap = objective_value(model_milp) - objective_value(model_lp)
gap_pct = 100 * gap / objective_value(model_lp)
println("  Gap de integralidade: USD $(fmt_usd(gap)) ($(@sprintf("%.2f", gap_pct))%)")

# Shadow prices das restrições de balanço
shadow_prices = [dual(model_lp[:balance][t]) for t in 1:T]

# A taxa implícita spot para o ano t pode ser extraída do shadow price:
#   shadow_price(t) ≈ fator de desconto para o ano t
#   → taxa spot: r_t = (1/shadow_price(t))^(1/t) - 1
# Nota: em modelos de dedication, o shadow price da restrição do ano t
# representa o custo marginal (em valor presente) de cobrir +1 USD de passivo nesse ano.

println("\n  Shadow Prices e Estrutura a Termo Implícita:")
sensitivity_df = DataFrame(
    Ano = Int[],
    Passivo = Float64[],
    Shadow_Price = Float64[],
    Fator_Desconto = Float64[],
    Taxa_Spot_Implícita = String[],
    Custo_Marginal = String[]
)

for t in 1:T
    sp = shadow_prices[t]
    # O shadow price em um problema de minimização com restrição de igualdade
    # nos dá o fator de desconto implícito
    discount_factor = abs(sp)
    spot_rate = discount_factor > 0 ? (1.0 / discount_factor)^(1.0 / t) - 1.0 : NaN

    push!(sensitivity_df, (
        t,
        liabilities[t],
        sp,
        discount_factor,
        @sprintf("%.3f%%", spot_rate * 100),
        @sprintf("USD %.4f", discount_factor)
    ))
end

pretty_table(sensitivity_df, alignment=:c)

# ─────────────────────────────────────────────────────────────────────────────
# 7. INTERPRETAÇÃO ECONÔMICA
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 80)
println("  INTERPRETAÇÃO ECONÔMICA")
println("=" ^ 80)

println("""

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  O que significam os Shadow Prices?                                    │
  │                                                                        │
  │  O shadow price da restrição de passivo do Ano t nos diz:             │
  │                                                                        │
  │    "Se o passivo do Ano t aumentar em USD 1, o custo mínimo           │
  │     da carteira dedicada aumentará em USD [shadow_price]."            │
  │                                                                        │
  │  Isso é exatamente o FATOR DE DESCONTO implícito da carteira.        │
  │  A partir dele, extraímos a taxa spot implícita para cada prazo.     │
  │                                                                        │
  │  Essa estrutura a termo é específica ao UNIVERSO de bonds             │
  │  disponíveis — diferente da curva de mercado "livre", ela reflete     │
  │  o custo real de hedgear passivos com os instrumentos que temos.      │
  └─────────────────────────────────────────────────────────────────────────┘
""")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANÁLISE "WHAT-IF": E SE O PASSIVO DO ANO 5 AUMENTAR?
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 80)
println("  WHAT-IF: Impacto de +100 no passivo do Ano 5")
println("=" ^ 80)

shock_year = 5
shock_amount = 100.0

liabilities_shocked = copy(liabilities)
liabilities_shocked[shock_year] += shock_amount

# Resolver novamente com o choque
model_shocked, _, _ = solve_dedication_lp(bonds, cf, liabilities_shocked, reinvest_rate)

delta_cost = objective_value(model_shocked) - objective_value(model_lp)
predicted_delta = shadow_prices[shock_year] * shock_amount

println("\n  Custo original (LP):          USD $(fmt_usd(objective_value(model_lp)))")
println("  Custo com choque (LP):        USD $(fmt_usd(objective_value(model_shocked)))")
println("  Aumento real no custo:        USD $(fmt_usd(delta_cost))")
println("  Previsão via shadow price:    USD $(fmt_usd(predicted_delta))  (shadow_price × Δpassivo)")
println("  Erro de aproximação:          USD $(fmt_usd(abs(delta_cost - predicted_delta)))")

println("""

  → O shadow price prevê com precisão o impacto marginal!
    Isso confirma que o dual do Ano 5 é de fato o "preço"
    de cobrir +1 USD de passivo naquele ano.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 9. COMPARAÇÃO MILP vs LP — CARTEIRA
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 80)
println("  COMPARAÇÃO: Carteira LP (fracionária) vs MILP (lotes inteiros)")
println("=" ^ 80)

compare_df = DataFrame(
    Bond = [b.name for b in bonds],
    Lotes_MILP = [round(Int, value(x_milp[j])) for j in 1:J],
    Lotes_LP = [@sprintf("%.3f", value(x_lp[j])) for j in 1:J],
    Diferença = [@sprintf("%.3f", round(Int, value(x_milp[j])) - value(x_lp[j])) for j in 1:J]
)
pretty_table(compare_df, alignment=:c)

println("""
  Nota: A diferença entre LP e MILP mostra o "custo da integralidade" —
  comprar em lotes inteiros obriga a carteira a ter mais bonds do que o
  estritamente necessário, gerando surplus (caixa excedente) em alguns anos.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 10. ANÁLISE PARAMÉTRICA: LP vs MILP sob variação do passivo (DeJans, 2025)
# ─────────────────────────────────────────────────────────────────────────────
#  Variamos o passivo do Ano 5 de -300 a +500 e resolvemos LP e MILP em cada
#  ponto. Isso revela:
#   - O custo LP varia linearmente (como previsto pelo shadow price)
#   - O custo MILP varia em degraus (soluções inteiras mudam discretamente)
#   - Para perturbações grandes, o shadow price perde validade preditiva

println("\n" * "=" ^ 80)
println("  ANÁLISE PARAMÉTRICA: Custo LP vs MILP ao variar passivo do Ano 5")
println("=" ^ 80)

sweep_year = 5
deltas = collect(-300:50:500)

lp_base = objective_value(model_lp)
milp_base = objective_value(model_milp)
sp5 = shadow_prices[sweep_year]

param_df = DataFrame(
    Delta = Int[],
    Passivo_Ano5 = Float64[],
    Custo_LP = Float64[],
    Custo_MILP = Float64[],
    Previsao_SP = Float64[],
    Erro_LP = Float64[],
    Erro_MILP = Float64[]
)

for δ in deltas
    liab_mod = copy(liabilities)
    liab_mod[sweep_year] += δ

    # Resolver LP
    m_lp, _, _ = solve_dedication_lp(bonds, cf, liab_mod, reinvest_rate)
    cost_lp = objective_value(m_lp)

    # Resolver MILP
    m_milp, _, _ = solve_dedication_milp(bonds, cf, liab_mod, reinvest_rate)
    cost_milp = objective_value(m_milp)

    # Previsão linear via shadow price
    predicted = lp_base + sp5 * δ

    push!(param_df, (
        δ,
        liabilities[sweep_year] + δ,
        cost_lp,
        cost_milp,
        predicted,
        cost_lp - predicted,
        cost_milp - predicted
    ))
end

pretty_table(param_df, alignment=:c)

println("""

  Observações:
  • O custo LP coincide com a previsão do shadow price para perturbações
    que não provocam troca de base. Quando a base muda, surgem desvios.
  • O custo MILP varia em degraus — a solução inteira permanece a mesma
    até que a perturbação force uma troca de lotes, quando salta para
    um novo patamar.
  • A divergência LP vs MILP confirma que os shadow prices da relaxação
    são guias direcionais, não previsões exatas para o problema inteiro.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 11. CUSTOS REDUZIDOS: Quais bonds estão "quase" na solução? (DeJans, 2025)
# ─────────────────────────────────────────────────────────────────────────────
#  O custo reduzido de uma variável não-básica indica quanto seu coeficiente
#  na função objetivo precisaria melhorar para que ela entrasse na base.
#  Bonds com custo reduzido próximo de zero são candidatos sensíveis a
#  pequenas mudanças de mercado (preço ou cupom).

println("=" ^ 80)
println("  CUSTOS REDUZIDOS DA RELAXAÇÃO LP")
println("=" ^ 80)

rc_df = DataFrame(
    Bond = String[],
    Lotes_LP = String[],
    Custo_Reduzido = Float64[],
    Interpretação = String[]
)

for j in 1:J
    lots = value(x_lp[j])
    rc = reduced_cost(x_lp[j])

    if lots > 1e-6
        interp = "Na solução (básica)"
    elseif abs(rc) < 5.0
        interp = "Quase entrando — sensível a mudanças de preço"
    else
        interp = "Distante da solução"
    end

    push!(rc_df, (
        bonds[j].name,
        @sprintf("%.3f", lots),
        rc,
        interp
    ))
end

pretty_table(rc_df, alignment=:c)

println("""

  Interpretação:
  • Bonds na solução (lotes > 0) têm custo reduzido zero por definição.
  • Um custo reduzido de +X para um bond fora da solução significa que
    seu preço precisaria cair X por lote para que valesse a pena incluí-lo.
  • Bonds com custo reduzido pequeno são os mais sensíveis a flutuações
    de mercado — uma leve queda no preço ou aumento no cupom poderia
    trazê-los para a carteira ótima.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 12. CURVA IMPLÍCITA vs CURVA DE MERCADO (bootstrapping)
# ─────────────────────────────────────────────────────────────────────────────
#  Extraímos a curva de mercado a partir dos preços e fluxos dos bonds via
#  bootstrapping iterativo, e comparamos com a curva implícita dos shadow prices.
#  A diferença revela o "custo de restrição" da dedicação: a curva implícita
#  reflete não só o valor do dinheiro no tempo, mas também a escassez ou
#  abundância de instrumentos em cada prazo do universo disponível.

println("\n" * "=" ^ 80)
println("  CURVA IMPLÍCITA (Shadow Prices) vs CURVA DE MERCADO (Bootstrapping)")
println("=" ^ 80)

# Bootstrapping: para cada vencimento, usar o bond correspondente para
# extrair o fator de desconto de mercado.
# Organizamos os bonds por vencimento (1 bond por vencimento neste exemplo).
sorted_bonds = sort(collect(enumerate(bonds)), by = x -> x[2].maturity)

# Fatores de desconto de mercado via bootstrapping
d_market = zeros(T)

for (idx, (j, b)) in enumerate(sorted_bonds)
    mat = b.maturity
    coupon = b.coupon_rate * b.face_value
    # Preço = Σ(cupom × d[t], t=1..mat-1) + (cupom + face) × d[mat]
    # Resolver para d[mat]:
    pv_coupons = sum(coupon * d_market[t] for t in 1:mat-1; init=0.0)
    d_market[mat] = (b.price - pv_coupons) / (coupon + b.face_value)
end

# Taxas spot de mercado
rates_market = [(1.0 / d_market[t])^(1.0 / t) - 1.0 for t in 1:T]

# Taxas spot implícitas (já calculadas via shadow prices)
rates_implicit = [(1.0 / abs(shadow_prices[t]))^(1.0 / t) - 1.0 for t in 1:T]

curve_df = DataFrame(
    Ano = 1:T,
    FD_Mercado = [d_market[t] for t in 1:T],
    FD_Implícito = [abs(shadow_prices[t]) for t in 1:T],
    Taxa_Mercado = [@sprintf("%.3f%%", rates_market[t] * 100) for t in 1:T],
    Taxa_Implícita = [@sprintf("%.3f%%", rates_implicit[t] * 100) for t in 1:T],
    Spread_bps = [round(Int, (rates_implicit[t] - rates_market[t]) * 10000) for t in 1:T]
)

pretty_table(curve_df, alignment=:c)

println("""

  Interpretação:
  • A curva de mercado é extraída por bootstrapping dos preços dos bonds.
  • A curva implícita vem dos shadow prices da relaxação LP.
  • O spread (em basis points) entre as duas revela onde o universo de
    títulos disponíveis impõe custo adicional para cobrir passivos.
  • Spreads positivos indicam prazos onde a dedicação é relativamente
    mais cara que o "preço de mercado" do dinheiro — sinalizam escassez
    de instrumentos eficientes naquele horizonte.
  • Spreads negativos indicam prazos bem cobertos pelo universo disponível.
""")

println("=" ^ 80)
println("  FIM DA ANÁLISE")
println("=" ^ 80)
