| combo | provenance | feasible | p99 TTFT (ms) | p99 TPOT (ms) | attainment | energy (J) | tok/J | P/D xfer p99 (ms) |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU-P + GPU-D | RTXPRO6000:vendor_spec[prefill] + RTXPRO6000:vendor_spec[decode] | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| GPU-P + NPU-D | RTXPRO6000:vendor_spec[prefill] + ASCEND-SIM-PROXY:placeholder[decode] | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| NPU-P + GPU-D | ASCEND-SIM-PROXY:placeholder[prefill] + RTXPRO6000:vendor_spec[decode] | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| NPU-P + NPU-D | ASCEND-SIM-PROXY:placeholder[prefill] + ASCEND-SIM-PROXY:placeholder[decode] | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| aggregated (GPU baseline) | RTXPRO6000:vendor_spec[aggregated] | no | 165.8 | 18.8 | 0.98 | 45,860 | 1.655 | - |
