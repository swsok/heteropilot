# `ClusterSpecV2` — the planner's cluster YAML, and what `schema_version: 2` adds

This is the **upper** of the fork's two schema layers. The planner reasons over
this YAML (`planner/inventory.py`) and *compiles down* to the untouched legacy
`configs/cluster/*.json` at simulation time. The lower layer is documented in
`docs/phase0_formats.md` §4 and in upstream's `docs/docs/reference/cluster-config.md`;
neither of those describes this file's schema, and they never have.

`schema_version` was added by STEP H1–H3 of the graph-search work order
(`WORK_ORDER_graph_search.md` in `swsok/heteropilot-graphsearch`, STEP H2;
deviation **D120**). Nothing
in the planner requires it yet — the graph-search driver in
`swsok/heteropilot-graphsearch` is its first consumer.

---

## The v1 guarantee

**A file with no `schema_version`, or `schema_version: 1`, means exactly what it
meant before H2.** Every committed cluster under `examples/clusters/` and
`experiments/configs/clusters/` is v1 and is unchanged.

That guarantee is enforced, not merely intended: a v1 file carrying any v2 field
at a non-default value is an `InventoryError` naming the field, rather than a
value that is quietly ignored.

```
InventoryError: node node0: schema_version 2 required for 'cpu_sockets';
this file declares schema_version 1, where that field does not exist
```

A field silently ignored is as bad as one silently honoured — only an error
distinguishes *"not supported here"* from *"accepted, and had no effect"*. That
is the same failure the metrics-schema digest exists to catch (D121), and the
same one D2 records for an unmeasurable constraint reading as satisfied.

### `bandwidth_gbps` is GB/s in both versions

The field is named `gbps` and has always held **GB/s** (`planner/topology.py`).
v2 does not reinterpret it. `bandwidth_unit` states the unit explicitly and may
select bits instead, but **only at `schema_version: 2`** — a v1 file setting it
to anything but `GB/s` is refused, because reinterpreting a committed number's
unit is how a 64 quietly becomes an 8.

---

## What v2 adds

### Vertices a v1 file cannot express

v1 knows two kinds of device: accelerators and NICs. A graph built from it
therefore cannot tell two GPUs sharing a CPU socket or a PCIe switch from two
that share nothing, and cannot see the uplink they both cross.

| Field | Scope | Meaning |
| --- | --- | --- |
| `nodes[].cpu_sockets[]` | node | `{id, numa_node}` |
| `nodes[].pcie_switches[]` | node | `{id, upstream}` — `upstream` names a `cpu_socket` or another `pcie_switch` **on the same node** |
| `net_switches[]` | **cluster** | `{id, ports}` |

A node's `accelerators`, `nics`, `cpu_sockets` and `pcie_switches` share **one id
namespace**, because a link endpoint is `<node>/<id>` whatever kind the id names.

**A `net_switch` belongs to no node**, so a link endpoint naming one carries no
`<node>/` prefix. That is how the two are told apart, and it is the only endpoint
form with no slash:

```yaml
links:
  - {id: l-nic0-sw0, src: node0/nic0, dst: sw0, type: ETHERNET, ...}
```

`Link.endpoints` reports such an endpoint as `("", "sw0")`. An empty node id is
rejected, so a switch can never be mistaken for a device on some node.

### Shared resources — capacity, not just a name

`Link.contention_group` (v1) says only *that* links share something. With no
capacity there is nothing to subtract a reservation from and no cut bound to
compute. `shared_resources` names the resource and gives it both:

```yaml
shared_resources:
  - {id: uplink-node0, kind: pcie_uplink, capacity: 10.0, unit: "GB/s",
     reserved: 6.0, node: node0, source: placeholder}
```

`kind` is one of `pcie_uplink | nic | switch_port | other`. `reserved` is held by
traffic this planner does not control and is subtracted before any bound is
computed — a bound over the full capacity is optimistic in a way no later
measurement can rescue. `reserved > capacity` is an error.

A link points at one with `shared_resource: <id>`. **Setting both
`contention_group` and `shared_resource` on one link is an error**: they name the
same thing, only one carries a capacity, so state one.

### Prices

| Field | Scope |
| --- | --- |
| `nodes[].accelerators[].price_per_hour_usd` | v2 cluster; overrides the profile's |
| `nodes[].host_price_per_hour_usd` | v2 cluster; charged in full to any plan touching the node |
| `price_per_hour_usd` + `price_source` | accelerator **profile** (not version-gated) |

A profile price with no `price_source` is refused. An unattributed price is a
made-up number (absolute rule 3), and a cost ranking built on one cannot be
defended. Unpriced is **None**, never zero and never the accelerator count: a
partial sum ranks the under-priced plan cheapest, so `MINIMIZE_COST_PER_HOUR`
declines to score a plan with any price missing (`pareto.can_score`, H1).

### Runtime capabilities

```yaml
runtime_capabilities:
  collectives: [all_reduce, p2p]
  max_world_size: 8
  kv_transfer: true
  source: placeholder
```

On the **profile**, and **absent means unstated, not unsupported**. A
compatibility check skips an unstated capability and records that it skipped it,
rather than rejecting a candidate for a fact nobody wrote down.

### Other link fields (v2 only)

`direction: bidir | src_to_dst` — v1 links are undirected and stay so.
`rdma`, `p2p` — tri-state; `None` is unstated.

### `snapshot_id`

Identifies the inventory reading the file was written from, so a plan can be
re-checked against a later one before it is deployed.

---

## Validation, in the order it runs

1. Duplicate `node` / `link` / `net_switch` / `shared_resource` ids — rejected.
2. Per node: duplicate ids **across** accelerators, NICs, sockets and switches;
   a `pcie_switch.upstream` that names nothing on that node.
3. `schema_version: 1` + any v2 field at a non-default value → names the field.
4. Node id empty or containing `/` — rejected.
5. Each link endpoint: with a `/`, it must name a known device; without one it
   must name a `net_switch`, and in a v1 file it is refused outright.
6. `shared_resource` must name an entry in `shared_resources`.
7. `shared_resources[].node`, when set, must be a node.

## A worked example

`tests/data/cluster_v2_min.yaml` is the smallest file exercising every vertex
kind: two nodes, a socket and a PCIe switch, one `net_switch`, one shared
resource with a reservation, and a `Gbit/s` link. Every value in it is
`source: placeholder` — it validates the schema and is not something to plan on.
