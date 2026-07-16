# Seasonal Profiles And Event Signals In The Environment

## Purpose

This note summarizes the new environment design for later paper writing, slide decks, and model explanation.

The key change is that the environment is no longer driven mainly by near-independent random risk draws. It now contains two predictable structure layers:

1. slow seasonal profiles
2. fast observable event signals

These two layers jointly drive next-step supply risk, transport risk, and price risk, which gives both heuristic planning and PPO policies forward-looking warning information.

## Core Causal Logic

The environment now follows the logic below:

$$
\text{Seasonal Profiles}_t,\ \text{Observable Signals}_t
\rightarrow
\text{Risk Pressure}_{t+1}
\rightarrow
\text{Supply / Transport / Price Risk}_{t+1}
$$

This means the model can observe warning information at week $t$ and use it to infer higher risk at week $t+1$, rather than waiting until disruption has already materialized.

## Seasonal Profiles

Four seasonal profiles are explicitly generated in the environment:

1. `planting_peak`: captures fertilizer demand season and replenishment pressure
2. `monsoon_pressure`: captures seasonal port and shipping disruption pressure
3. `maintenance_cycle`: captures elevated probability of planned or semi-planned upstream maintenance
4. `winter_energy`: captures energy-cost and production-cost pressure in colder periods

These are smooth time-varying profiles rather than one-off random shocks.

## Observable Event Signals

The environment also generates observable event signals that work as short-term warning indicators:

1. `mine_accident`
2. `sanctions_tightening`
3. `maintenance_warning`
4. `port_congestion_alert`
5. `storm_alert`
6. `rail_disruption_alert`
7. `energy_shock`
8. `demand_surge_signal`

These signals have persistence and duration. They are not i.i.d. weekly Bernoulli noise.

## How Signals Affect The Three Risks

### 1. Supply Risk

Supply risk is mainly driven by:

1. `mine_accident`
2. `sanctions_tightening`
3. `maintenance_warning`
4. `energy_shock`

Example interpretation:

If a major mine accident or sanctions escalation is observed in week $t$, then the probability of high supply disruption in week $t+1$ increases substantially.

### 2. Transport Risk

Transport risk is mainly driven by:

1. `port_congestion_alert`
2. `storm_alert`
3. `rail_disruption_alert`
4. part of `sanctions_tightening`

Example interpretation:

If a port congestion alert or storm warning appears in week $t$, then node-level disruption probability and route delay risk rise in week $t+1$.

### 3. Price Risk

Price risk is mainly driven by:

1. `energy_shock`
2. `demand_surge_signal`
3. `mine_accident`
4. `sanctions_tightening`
5. spillover from supply pressure and transport pressure

Example interpretation:

If the origin market experiences an accident and energy shock while domestic replenishment demand is strong, then spot prices are more likely to jump in the next step.

## Why This Helps Control Policies

The control stack benefits when current observations contain forward-looking information about future risk.

Under the new environment design, the policy can use:

1. seasonal profiles as slow-moving background priors
2. event signals as immediate warning evidence
3. current probabilities and multipliers as already-materialized stress evidence

to respond before disruptions fully materialize, rather than reacting only after realized failures are already visible in costs, backlog, or capacity.

## Managerial Interpretation

For a focal importer-distributor such as Anhui Huilong, the new environment means decisions can now react to signals before disruption fully realizes.

Typical uses include:

1. increase safety stock when supply warning signals intensify
2. switch route allocation when transport warning signals intensify
3. hedge or pre-buy when price warning signals intensify

This is closer to real procurement and logistics practice, where firms act on warnings, not only on realized failures.
