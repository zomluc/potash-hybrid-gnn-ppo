# PPO Observation State And Action Space

## Purpose

This document lists the full PPO observation state and action space used in the current supply-chain environment, together with a plain-language explanation of what each quantity means in actual supply-chain decision making.

The implementation described here matches the current code in `rl_environment.py` and `simulation_types.py`.

## Supply Chain Network Structure

Before listing the observation and action variables, it is useful to clarify the physical supply-chain network represented by the environment.

The network is a global imported potash supply chain centered on Anhui Huilong. Upstream potash is procured from several foreign suppliers, then transported through Chinese import / land gateway nodes, and finally enters the focal firm's inventory system for fertilizer processing and downstream distribution.

### Core Enterprise

The focal enterprise is Anhui Huilong. In the model it is represented by a single `focal_firm` node.

Its role is:

1. imported potash purchaser
2. fertilizer manufacturer
3. downstream distributor

So in this environment, the manufacturer and distributor are not modeled as two separate nodes. They are merged into one focal decision-making node.

### Node Categories

The network contains three types of nodes:

1. upstream supplier nodes
2. import / transfer / land-gateway nodes
3. the focal manufacturer-distributor node

There is no separate explicit downstream distributor-node layer in the graph. Downstream sales demand is aggregated into the focal firm's demand process.

### Supplier Nodes And Locations

| Node | Supplier | Geographic location | Role in the network |
|---|---|---|---|
| `supplier_LA` | Laos | Laos | Short-lead upstream potash source, mainly linked to the China land corridor |
| `supplier_CA` | Canada | Canada | Large overseas seaborne supply source |
| `supplier_RU` | Russia | Russia | Overseas supplier with both sea and land routing options |
| `supplier_JO` | Jordan | Jordan | Middle East seaborne supplier linked to the Aqaba--Lianyungang corridor |

### Import / Gateway Nodes And Locations

| Node | Full name | Geographic location | Role in the network |
|---|---|---|---|
| `port_ZJ` | Zhanjiang | Guangdong, China | Major seaborne import and transfer gateway |
| `port_FCG` | Fangchenggang | Guangxi, China | Alternative seaborne import gateway |
| `port_MH` | Mohan | Yunnan, China | Southwest land / rail gateway for Laos corridor |
| `port_LYG` | Lianyungang | Jiangsu, China | Seaborne receiving gateway for Jordan potash |
| `port_MZL` | Manzhouli | Inner Mongolia, China | Northern land gateway |

### Focal Firm Node And Location

| Node | Enterprise | Geographic location | Role in the network |
|---|---|---|---|
| `focal_firm` | Anhui Huilong | Anhui, China | Core enterprise, responsible for procurement, inventory, processing, and downstream distribution |

### Route Links And Transport Modes

The graph edges represent feasible procurement and transport corridors from each supplier to a Chinese gateway node.

| Route | From | To | Transport mode | Meaning |
|---|---|---|---|---|
| `LA_MH` | Laos | Mohan | Rail | Main Laos-to-China land corridor |
| `CA_ZJ` | Canada | Zhanjiang | Sea | Main Canadian seaborne route |
| `CA_FCG` | Canada | Fangchenggang | Sea | Backup Canadian seaborne route |
| `RU_ZJ` | Russia | Zhanjiang | Sea | Main Russian seaborne route |
| `RU_MZL` | Russia | Manzhouli | Land | Backup Russian land route |
| `JO_LYG` | Jordan | Lianyungang | Sea | Main Jordan-to-Lianyungang seaborne route |
| `JO_ZJ` | Jordan | Zhanjiang | Sea | Backup Jordan-to-China seaborne route |

### Network Interpretation

This network should be read as a procurement and inbound-logistics network, not as a full end-customer distribution graph.

Its key modeling purpose is to capture:

1. multi-origin sourcing decisions
2. multi-route transport switching decisions
3. gateway congestion and disruption exposure
4. inventory and replenishment decisions at the focal firm
5. price and hedging decisions under uncertainty

With this structure in mind, the observation variables below describe the current state of suppliers, gateways, routes, the focal firm, and exogenous risk signals; the action variables describe how the focal firm adjusts sourcing, safety stock, routing, reserve capacity, and hedging.

## 1. Observation Space Overview

The environment observation is a dictionary with six parts:

| Observation key | Shape | Main use |
|---|---:|---|
| `node_features` | `(10, 8)` | Graph node state for suppliers, ports, and focal firm |
| `edge_features` | `(7, 7)` | Graph edge state for routes |
| `edge_index` | `(2, 7)` | Fixed route connectivity graph |
| `global_features` | `(15,)` | Actor-side compact global summary |
| `global_feature_history` | `(4, 15)` by default | Recent actor-side global summaries for temporal risk context |
| `flat_observation` | `(19,)` | Critic-side flat summary |

### Node Order

The 10 nodes are ordered as follows:

| Node index | Node name | Meaning |
|---:|---|---|
| 0 | `supplier_LA` | Laos supply node |
| 1 | `supplier_CA` | Canada supply node |
| 2 | `supplier_RU` | Russia supply node |
| 3 | `supplier_JO` | Jordan supply node |
| 4 | `port_ZJ` | Zhanjiang import / transfer node |
| 5 | `port_FCG` | Fangchenggang import / transfer node |
| 6 | `port_MH` | Mohan land gateway node |
| 7 | `port_LYG` | Lianyungang receiving gateway node |
| 8 | `port_MZL` | Manzhouli land gateway node |
| 9 | `focal_firm` | Anhui Huilong as focal manufacturer-distributor |

### Edge Order

The 7 route edges are ordered as follows:

| Edge index | Route name | Physical meaning |
|---:|---|---|
| 0 | `LA_MH` | Laos to Mohan by rail |
| 1 | `CA_ZJ` | Canada to Zhanjiang by sea |
| 2 | `CA_FCG` | Canada to Fangchenggang by sea |
| 3 | `RU_ZJ` | Russia to Zhanjiang by sea |
| 4 | `RU_MZL` | Russia to Manzhouli by land |
| 5 | `JO_LYG` | Jordan to Lianyungang by sea |
| 6 | `JO_ZJ` | Jordan to Zhanjiang by sea |

## 2. Node Features

Each node has 8 features in total, but these 8 dimensions are not 8 homogeneous business variables.

They are composed as:

1. 3 node-type indicator dimensions
2. 5 node-specific operational dimensions

So if one counts only the operational variables and ignores the type encoding, then each node is indeed described by 5 business features rather than 8 business features.

The interpretation depends on node type.

### Shared Type Indicators

| Feature index | Field | Meaning |
|---:|---|---|
| 0 | `type_1` | Supplier indicator |
| 1 | `type_2` | Port / gateway indicator |
| 2 | `type_3` | Focal-firm indicator |

Exactly one of these three entries is `1` for each node.

These three dimensions are included because the graph encoder uses one shared parameterization across suppliers, ports, and the focal firm. Without node-type indicators, the model would have to infer node category only indirectly from feature scale and position, which is unnecessarily hard and unstable.

### Supplier Node Features

For `supplier_LA`, `supplier_CA`, `supplier_RU`, `supplier_JO`:

| Feature index | Field | Explanation |
|---:|---|---|
| 3 | `weekly_capacity_norm` | Supplier weekly capacity, normalized by `50000` tons |
| 4 | `spot_price_norm` | Current supplier spot price, normalized by `500` |
| 5 | `supply_prob` | Current realized supply disruption probability at this origin |
| 6 | `supply_multiplier` | Current realized effective supply multiplier at this origin; lower means stronger disruption |
| 7 | `last_supplier_weight` | Last policy's procurement weight assigned to this supplier |

Operational meaning:

`last_supplier_weight` is action-dependent exposure. It tells the agent how much of its last procurement plan was relying on that supplier.

So a supplier node is represented as:

1. 3 shared type-indicator dimensions
2. 5 supplier-specific operational dimensions

for a total of 8 dimensions.

### Port / Gateway Node Features

For `port_ZJ`, `port_FCG`, `port_MH`, `port_LYG`, `port_MZL`:

| Feature index | Field | Explanation |
|---:|---|---|
| 3 | `incoming_capacity_norm` | Sum of route base capacities into the node, normalized by `20000` tons |
| 4 | `avg_route_cost_norm` | Average inbound route freight plus handling cost, normalized by `120` |
| 5 | `node_prob` | Current realized disruption probability at this node |
| 6 | `node_multiplier` | Current realized effective node capacity multiplier; lower means stronger congestion / disruption |
| 7 | `last_node_exposure` | Exposure share induced by last policy after route allocation |

Operational meaning:

`last_node_exposure` tells the agent how dependent its most recent procurement plan is on that node.

More detailed interpretation of the four most important port-side variables:

1. `incoming_capacity_norm`
	This is the static inbound handling capacity proxy of the node. In the code it is computed as the sum of the base capacities of all routes entering that node, then normalized by `20000` tons.
	It answers the question: how much upstream flow can this gateway structurally absorb if there is no severe disruption?

2. `node_prob`
	This is the current disruption probability of the node. It reflects how likely the gateway is to suffer congestion, disruption, or operational stress at the current step.
	It is influenced by seasonal pressure and warning signals such as congestion alerts, storm alerts, and rail disruption alerts.

3. `node_multiplier`
	This is the realized effective capacity multiplier of the node. It captures how much usable gateway capacity remains after current disruption effects are applied.
	A value near `1.0` means the node is almost normal; a much lower value means gateway throughput has already been compressed by disruption.

4. `last_node_exposure`
	This is the policy-induced exposure of the node under the previous step's routing decision. It is the share of total planned procurement flow that would pass through that gateway.
	It answers a different question from `node_prob`: not how risky the node is, but how dependent the current sourcing plan is on that node.

Taken together, these four variables describe:

1. structural capacity at the node
2. current risk intensity at the node
3. realized capacity loss at the node
4. dependence of the current policy on the node

So a port / gateway node is represented as:

1. 3 shared type-indicator dimensions
2. 5 port-specific operational dimensions

for a total of 8 dimensions.

### Focal Firm Node Features

For `focal_firm`:

| Feature index | Field | Explanation |
|---:|---|---|
| 3 | `inventory_norm` | On-hand inventory, normalized by `50000` tons |
| 4 | `pipeline_qty_norm` | Total in-transit quantity, normalized by `50000` tons |
| 5 | `backlog_norm` | Unmet demand carried into future, normalized by `10000` tons |
| 6 | `inbound_1w_norm` | Shipments arriving within 1 week, normalized by `20000` tons |
| 7 | `current_demand_norm` | Current weekly demand, normalized by `15000` tons |

So the focal-firm node is represented as:

1. 3 shared type-indicator dimensions
2. 5 focal-firm operational dimensions

for a total of 8 dimensions.

### Are All Node Features Necessary?

Broadly, yes, but they play different roles.

#### Structurally Necessary Features

These are the features that tell the policy what kind of object it is looking at and what physical constraints exist:

1. node-type indicators
2. supplier capacity
3. port incoming capacity
4. route- or node-side cost proxies
5. inventory / pipeline / backlog / demand at the focal firm

These features are necessary because procurement and routing decisions must respect who the node is and what constraints it faces.

#### Risk-Relevant Features

These are the features that tell the policy what uncertainty is currently materializing:

1. `supply_prob`
2. `supply_multiplier`
3. `node_prob`
4. `node_multiplier`

These are necessary because sourcing and routing decisions should respond to realized disruption pressure.

#### Action-Coupled Exposure Features

These are the features that connect current state to previous decisions:

1. `last_supplier_weight`
2. `last_node_exposure`

These are especially useful because they tell the policy not only what the environment looks like, but also where the current plan is exposed. Without them, the network would know that a node is risky, but not how much the current sourcing pattern depends on that node.

#### Could Some Features Be Removed?

Possibly, but that would be an ablation question rather than a documentation question.

The most defensible current view is:

1. type indicators are necessary for heterogeneous-node graph encoding
2. operational quantities are necessary for inventory and replenishment decisions
3. realized risk quantities are necessary for responsive sourcing and routing
4. exposure quantities are necessary for linking risk to the current policy footprint

So the current design is not arbitrary. It is intended to make the observation state decision-relevant rather than merely descriptive.

## 3. Edge Features

Each route edge has 7 features.

| Feature index | Field | Explanation |
|---:|---|---|
| 0 | `mode_sea` | Sea transport indicator |
| 1 | `mode_rail` | Rail transport indicator |
| 2 | `mode_land` | Land transport indicator |
| 3 | `lead_time_norm` | Route lead time in weeks, normalized by `6` |
| 4 | `route_cost_norm` | Freight plus handling cost, normalized by `120` |
| 5 | `route_risk` | Current route risk proxy, defined as `max(supplier risk, node risk)` |
| 6 | `last_route_exposure` | Exposure share of this route implied by the last policy |

Operational meaning:

`last_route_exposure` tells the actor how much of recent procurement flow is currently routed through each corridor.

More detailed interpretation of the edge variables:

1. `mode_sea`, `mode_rail`, `mode_land`
	These three entries are transport-mode indicators. They let the graph model distinguish routes with very different operational characteristics, such as long seaborne routes versus short land corridors.

2. `lead_time_norm`
	This is the baseline transport lead time of the route, normalized by `6` weeks.
	It matters because route switching is not only a cost decision; it is also a replenishment timing decision.

3. `route_cost_norm`
	This is the sum of base freight and handling cost, normalized by `120`.
	It represents the direct logistics cost level of using the corridor.

4. `route_risk`
	This is a simple current-step route risk proxy defined as `max(supplier risk, node risk)`.
	It compresses upstream origin disruption and downstream gateway disruption into one corridor-level stress indicator.

5. `last_route_exposure`
	This is the share of planned procurement flow assigned to the route under the previous decision.
	It is action-coupled and tells the actor where its current logistics footprint is concentrated.

So the edge features jointly describe:

1. what kind of corridor the route is
2. how slow it is
3. how expensive it is
4. how risky it currently is
5. how much the current plan depends on it

## 4. Edge Index

`edge_index` is fixed graph structure, not a stochastic state variable.

| Route | Source node | Destination node |
|---|---|---|
| `LA_MH` | `supplier_LA` | `port_MH` |
| `CA_ZJ` | `supplier_CA` | `port_ZJ` |
| `CA_FCG` | `supplier_CA` | `port_FCG` |
| `RU_ZJ` | `supplier_RU` | `port_ZJ` |
| `RU_MZL` | `supplier_RU` | `port_MZL` |
| `JO_LYG` | `supplier_JO` | `port_LYG` |
| `JO_ZJ` | `supplier_JO` | `port_ZJ` |

This matrix does not contain business magnitudes by itself. Its role is to tell the graph neural network which supplier nodes are connected to which gateway nodes.

In other words:

1. `node_features` say what each node currently looks like
2. `edge_features` say what each transport corridor currently looks like
3. `edge_index` says which nodes are linked by which corridor

Without `edge_index`, the graph encoder would not know that, for example, Canada is linked to both Zhanjiang and Fangchenggang, or that Russia has both a sea path and a land path.

## 5. Global Features

`global_features` are the actor-side compact summary vector.

### Base 15 Global Features

| Index | Field | Explanation |
|---:|---|---|
| 0 | `inventory_norm` | On-hand inventory / `50000` |
| 1 | `backlog_norm` | Backlog / `10000` |
| 2 | `pipeline_qty_norm` | Total in-transit quantity / `50000` |
| 3 | `inbound_1w_norm` | Near-term inbound quantity / `20000` |
| 4 | `inbound_2_3w_norm` | Inbound quantity arriving in 2 to 3 weeks / `30000` |
| 5 | `current_demand_norm` | Current demand divided by the environment's dynamic demand normalization scale |
| 6 | `basket_spot_norm` | Average supplier spot basket price / `500` |
| 7 | `supplier_concentration` | Herfindahl-style concentration of supplier weights, sum of squared shares |
| 8 | `planting_peak` | Seasonal fertilizer demand profile intensity |
| 9 | `monsoon_pressure` | Seasonal port / shipping disruption pressure |
| 10 | `maintenance_cycle` | Seasonal upstream maintenance pressure |
| 11 | `winter_energy` | Seasonal energy-cost pressure |
| 12 | `supply_warning` | Aggregated supply warning intensity from event signals |
| 13 | `transport_warning` | Aggregated transport warning intensity from event signals |
| 14 | `price_warning` | Aggregated price warning intensity from event signals |

### How The Warning Signals Are Generated

The three warning-summary fields above:

1. `supply_warning`
2. `transport_warning`
3. `price_warning`

are aggregated summaries of a deeper layer of raw observable event signals generated inside the simulator.

The raw signals are:

1. `mine_accident`
2. `sanctions_tightening`
3. `maintenance_warning`
4. `port_congestion_alert`
5. `storm_alert`
6. `rail_disruption_alert`
7. `energy_shock`
8. `demand_surge_signal`

These raw signals should be understood as a stochastic process, not as independent weekly Bernoulli draws with one fixed probability.

More precisely, each raw signal follows an event-driven random process with four components:

1. event start probability
2. random duration
3. random intensity
4. temporal decay and cross-signal interaction

In each week, if a signal is currently inactive, the simulator checks whether a new event starts. The start probability is not constant. It is based on:

1. a signal-specific base probability
2. a seasonal bump for some weeks of the year
3. a seasonal-profile adjustment such as `planting_peak`, `monsoon_pressure`, `maintenance_cycle`, or `winter_energy`

So the signal-generation logic is better described as:

$$
p_{start,t} = \text{base probability} + \text{seasonal bump}_t + \text{seasonal profile adjustment}_t
$$

If an event starts, it does not disappear immediately in the next week. Instead:

1. a random duration is drawn
2. a random initial amplitude is drawn
3. the signal remains active for several weeks
4. the amplitude gradually decays across weeks
5. with small probability, the amplitude can temporarily intensify again during the active spell

This is why the warning layer has persistence and predictive value.

In addition, the raw signals are not fully independent. The simulator imposes several causal couplings, for example:

1. stronger `storm_alert` increases `port_congestion_alert`
2. stronger `sanctions_tightening` increases `rail_disruption_alert`
3. stronger `sanctions_tightening` also increases `energy_shock`

The warning summaries in the PPO observation are therefore compact early-warning aggregates rather than realized disruption outcomes. Conceptually, the information flow is:

$$
	ext{raw event signals} \rightarrow \text{warning summaries} \rightarrow \text{risk pressures} \rightarrow \text{realized disruption probabilities and outcomes}
$$

The warning summaries are intentionally upstream signals. They provide early-warning context before disruption probabilities and realized operational losses have fully materialized.

### Demand Seasonality Under The Current 52-Week Setup

The demand process now uses a stronger seasonal coefficient schedule than before.

Its design target is:

1. baseline seasonality coefficient near `1.0`
2. peak-season coefficient up to about `2.5`
3. trough-season coefficient down to about `0.5`
4. longer high-demand windows rather than only brief spikes

The weekly demand path is therefore driven by three layers:

1. a base average weekly demand level
2. a seasonal coefficient that varies across the year
3. random demand noise plus occasional `demand_surge_signal` uplift

Under the current 52-week default setting, the broad calendar interpretation is approximately:

| Approximate weeks | Demand regime | Interpretation |
|---|---|---|
| 1 to 2 | trough / pre-season | low application demand before the main spring window opens |
| 2 to 20 | spring main peak | extended spring fertilization window with sustained high demand |
| 19 to 25 | spring-summer shoulder | still elevated demand, but below the spring peak crest |
| 26 to 27 | mid-year transition | relatively softer demand between major application windows |
| 28 to 49 | autumn secondary peak | long autumn replenishment and application window with high demand |
| 50 to 52 | year-end support tail | residual elevated demand before returning toward trough conditions |

These week bands are approximate because the implementation scales legacy 30-week seasonal windows onto the active episode horizon.

The practical consequence is that `current_demand_norm` can now spend longer periods well above its old mid-season level, and peak weeks can remain elevated for a materially larger portion of the episode.

### Typical Raw Signal Starts Per Episode

Under the current default 52-week episode setting, the raw signals still start only a limited number of times per episode rather than appearing independently every week.

The exact average number of starts depends on the chosen episode horizon, so changing the default from 30 weeks to 52 weeks means these empirical counts should be re-benchmarked if you need precise frequency estimates.

As a rough order-of-magnitude reference from the earlier 30-week calibration, the simulator produced the following typical counts:

| Raw signal | Illustrative starts in one episode | Typical interpretation |
|---|---:|---|
| `mine_accident` | about `1` time on average | low-frequency upstream shock |
| `sanctions_tightening` | about `1` time on average | low-frequency geopolitical shock |
| `maintenance_warning` | about `2` times on average | recurring upstream maintenance pressure |
| `port_congestion_alert` | about `3` times on average | relatively frequent logistics warning |
| `storm_alert` | about `2` times on average | seasonal short transport shock |
| `rail_disruption_alert` | about `2` times on average | intermittent land-corridor warning |
| `energy_shock` | about `2` times on average | recurring energy-cost shock |
| `demand_surge_signal` | about `2` times on average | recurring demand-side surge warning |

These are not hard-coded counts. They are emergent outcomes of the stochastic generation process.

So if one asks whether the raw event layer is random, the answer is yes; but it is a structured random process with persistence, seasonality, and coupling, rather than a memoryless dice roll each week.

### Temporal Global History For The GNN Actor

In addition to the current-step `global_features`, the environment now exposes `global_feature_history`.

By default its shape is `(4, 15)`, meaning the actor receives a short rolling window of the most recent four global-summary vectors, including the current step.

Its purpose is to help the GNN actor model:

1. persistence of warning signals
2. multi-week buildup of supply and transport stress
3. short-term temporal trends in inventory, backlog, inbound flow, and demand

At the beginning of an episode, the unavailable earlier slots are zero-padded.

This temporal history is primarily used by the enhanced `gnn_ppo` actor. The critic still relies on the current-step `flat_observation` summary.

## 6. Flat Observation

`flat_observation` is the critic-side flat summary vector. It overlaps with `global_features`, but is designed for value estimation rather than graph message passing.

### Base 19 Flat Features

| Index | Field | Explanation |
|---:|---|---|
| 0 | `inventory_norm` | On-hand inventory / `50000` |
| 1 | `backlog_norm` | Backlog / `10000` |
| 2 | `pipeline_qty_norm` | Total in-transit quantity / `50000` |
| 3 | `inbound_1w_norm` | Near-term inbound quantity / `20000` |
| 4 | `inbound_2_3w_norm` | Inbound quantity in 2 to 3 weeks / `30000` |
| 5 | `current_demand_norm` | Current demand / `15000` |
| 6 | `cumulative_shortage_norm` | Cumulative shortage to date / `50000` |
| 7 | `basket_spot_norm` | Current average spot basket price / `500` |
| 8 | `max_supply_prob` | Max current supplier disruption probability |
| 9 | `min_supply_multiplier` | Worst current supplier availability multiplier |
| 10 | `max_node_prob` | Max current node disruption probability |
| 11 | `min_node_multiplier` | Worst current node capacity multiplier |
| 12 | `planting_peak` | Seasonal demand-profile intensity |
| 13 | `monsoon_pressure` | Seasonal transport-disruption profile |
| 14 | `maintenance_cycle` | Seasonal maintenance profile |
| 15 | `winter_energy` | Seasonal energy-cost profile |
| 16 | `supply_warning` | Aggregated supply warning signal |
| 17 | `transport_warning` | Aggregated transport warning signal |
| 18 | `price_warning` | Aggregated price warning signal |

### Why `flat_observation` Is Similar To But Not The Same As `global_features`

The two vectors describe the same environment state, so some overlap is intentional. However, they are not designed for the same network component.

`global_features` is the compact graph-level summary used by the actor, while `flat_observation` is the compact value-estimation summary used by the critic.

This distinction matters because the actor already receives rich graph-structured information through:

1. `node_features`
2. `edge_features`
3. `edge_index`

So `global_features` does not need to repeat every important supplier-level and node-level risk statistic in compressed form. Its role is mainly to provide additional global context that is useful alongside the graph, such as:

1. inventory and pipeline status
2. aggregate demand and basket price conditions
3. last policy structure such as `supplier_concentration`
4. exogenous seasonal and warning summaries

By contrast, the critic does not operate on the graph structure directly. It receives one flat vector and must estimate the state's long-run value from that vector alone.

For that reason, `flat_observation` includes several quantities that are especially useful for value estimation but are not as necessary in `global_features`, such as:

1. `cumulative_shortage_norm`
2. `max_supply_prob`
3. `min_supply_multiplier`
4. `max_node_prob`
5. `min_node_multiplier`

These fields summarize how bad the current situation already is and how severe the worst current bottlenecks are. They are strong value-relevant signals for the critic because future return is highly sensitive to cumulative shortage and worst-case disruption pressure.

So the main design logic is:

1. actor side: combine graph detail with a short global context vector
2. critic side: use a flatter but more value-oriented summary vector

This is also why `global_features` contains strategy-structure summaries such as `supplier_concentration`, while `flat_observation` instead gives more emphasis to worst-case current risk and accumulated shortage.

In short, the overlap exists because both vectors describe the same underlying state, but the differences exist because the actor and critic solve different subproblems:

1. the actor needs decision-oriented context on top of graph structure
2. the critic needs a compact summary that is strongly predictive of future total reward

## 7. Action Space Overview

The PPO action is a continuous vector of length 9:

| Raw action index | Decoding rule | Output policy field |
|---:|---|---|
| 0 to 3 | Softmax over four logits | `supplier_weights` |
| 4 | `1.0 + 5.0 * sigmoid(a4)` | `safety_stock_weeks` |
| 5 | `0.50 * sigmoid(a5)` | `reserve_supplier_ratio` |
| 6 | `0.80 * sigmoid(a6)` | `backup_route_ratio` |
| 7 | `0.50 * sigmoid(a7)` | `transport_reserve_ratio` |
| 8 | `0.95 * sigmoid(a8)` | `futures_hedge_ratio` |

## 8. Detailed Action Mapping

### Supplier Allocation Actions

Action dimensions `0` to `3` are converted by a softmax into procurement shares across the four origins.

| Raw action dim | Policy field | Supply-chain meaning |
|---:|---|---|
| 0 | `supplier_weights[LA]` | Share of procurement budget / volume assigned to Laos |
| 1 | `supplier_weights[CA]` | Share assigned to Canada |
| 2 | `supplier_weights[RU]` | Share assigned to Russia |
| 3 | `supplier_weights[JO]` | Share assigned to Jordan |

These four shares always sum to 1.

### Scalar Policy Actions

| Raw action dim | Output range | Policy field | Actual operational decision |
|---:|---:|---|---|
| 4 | `[1, 6]` weeks | `safety_stock_weeks` | Target safety-stock coverage used in replenishment planning |
| 5 | `[0, 0.5]` | `reserve_supplier_ratio` | Extra supplier-side reserve commitment used to buffer origin disruptions |
| 6 | `[0, 0.8]` | `backup_route_ratio` | Portion of flow shifted from primary routes to backup corridors |
| 7 | `[0, 0.5]` | `transport_reserve_ratio` | Extra transport capacity reserve retained to absorb node / route disruption |
| 8 | `[0, 0.95]` | `futures_hedge_ratio` | Share of procured quantity hedged through futures |

### Spot Procurement And `futures_hedge_ratio`

The current simulator now assumes unified spot procurement for physical purchasing.

That means the purchase price of supplier `i` is simply the current spot price:

$$
P_i = P_i^{spot}
$$

So physical procurement no longer has a separate contract-price mix decision. The remaining price-risk control is `futures_hedge_ratio`, which is a financial hedging decision rather than a physical purchase-pricing decision.

`futures_hedge_ratio` determines what share of the procured quantity is hedged through a futures position that is held until the expected arrival week of that procured batch.

Operationally:

1. physical procurement is still executed at current spot prices
2. after procurement is executed, part of that procured quantity can be hedged financially
3. the hedge is attached to the procured shipment and settles at that shipment's expected arrival week
4. the hedge does not change the physical purchase price directly; it adds hedge P&L and hedge transaction cost

In the current environment, the futures price is not modeled through a full term-structure model. Instead it is generated as a spot-linked synthetic hedge price:

$$
F_t = S_t + 6 + 8 \cdot \text{price pressure}_t + \varepsilon_t
$$

where:

1. $S_t$ is the current spot price
2. `price pressure` is the current price-risk pressure from the environment
3. $\varepsilon_t$ is an additional noise term

So the futures price should be interpreted as a spot-related hedge instrument with a positive basis and a higher premium in high-volatility periods.

The hedge quantity is:

$$
Q_t^{hedge} = \text{procured quantity}_t \cdot \text{futures\_hedge\_ratio}
$$

For a hedged shipment from supplier $i$, the hedge is opened at order time and settled at the shipment's arrival week. The hedge P&L is:

$$
	ext{hedge pnl}_{i} = Q_{i}^{hedge} \cdot \left(F_{i,t}^{locked} - S_{i,\tau}^{spot}\right)
$$

where:

1. $t$ is the order week
2. $\tau$ is the expected arrival week of that shipment
3. $F_{i,t}^{locked}$ is the supplier-specific futures price locked at order time
4. $S_{i,\tau}^{spot}$ is the supplier-specific spot price at settlement

This means the hedge gains when market price falls after procurement, which is consistent with the objective of protecting downstream gross margin after upstream purchase has already been priced.

The simulator also charges a hedge transaction cost proportional to hedged quantity.

## 9. How Actions Become Actual Supply-Chain Moves

The decoded `Policy` does not move inventory directly. Instead it shapes the weekly procurement and routing logic.

| Policy field | Downstream effect in the simulator |
|---|---|
| `supplier_weights` | Determines how planned procurement volume is split across Laos, Canada, Russia, and Jordan |
| `safety_stock_weeks` | Raises or lowers the replenishment target inventory position |
| `reserve_supplier_ratio` | Expands effective available supplier capacity at added reserve cost |
| `backup_route_ratio` | Diverts part of Canada, Russia, and Jordan flows onto backup routes |
| `transport_reserve_ratio` | Expands effective route capacity at added reserve cost |
| `futures_hedge_ratio` | Creates arrival-matched futures hedge positions on a portion of procured quantity to protect gross margin against post-procurement price declines |

## 10. Route-Level Meaning Of Backup Decisions

`backup_route_ratio` has different physical meaning by origin:

| Supplier | Primary route | Backup route | Interpretation |
|---|---|---|---|
| Laos | `LA_MH` | None | Laos has no backup route in the current model |
| Canada | `CA_ZJ` | `CA_FCG` | Some Canadian seaborne flow is diverted from Zhanjiang to Fangchenggang |
| Russia | `RU_ZJ` | `RU_MZL` | Some Russian flow is shifted from sea-to-Zhanjiang to land via Manzhouli |
| Jordan | `JO_LYG` | `JO_ZJ` | Some Jordan flow is shifted from the Lianyungang receiving route to the backup Zhanjiang seaborne route |

## 11. Interpretation Notes

1. `node_features` and `edge_features` are the graph-structured operational state.
2. `global_features` are the actor's current-step compact macro summary.
3. `global_feature_history` gives the GNN actor a short temporal window of recent risk and operating conditions.
4. `flat_observation` is the critic's compact valuation state.
5. Seasonal profiles and event-warning summaries are part of the observation for both policy families.
6. The enhanced `gnn_ppo` actor now uses risk-conditioned message passing, attention-based node/route aggregation, and temporal context from recent global summaries.
7. Physical procurement is now priced uniformly at current spot prices; `futures_hedge_ratio` is the remaining price-risk control.
8. Futures hedge positions now settle at shipment arrival rather than one week later, so hedge horizon is aligned with procurement lead time.
