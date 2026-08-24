# main tests

test_flows/test_npe/test_embeddings/conftest

Construction and shape tests for all the embeddings, flows and NPE constructor.
They do **not** check that anything trains well — they check that everything at least
doesn't break when put together, which is where most of the issues have come from... (context
width mismatches, block-layout disagreements, batch-1 failures, silently
transposed or discarded data).

```bash
pytest tests -v                       # all of it, ~1 min on CPU
pytest tests/test_npe.py -v           # just the mix-and-match matrix
pytest tests -k FDPSD                 # one embedding across all flows
pytest tests -x -q                    # stop at the first failure
```

Configs come from `conftest.py` and are deliberately tiny (0.25 s at 256 Hz,
2 transforms, 16 hidden), so the full matrix runs on a laptop CPU.

What each file covers:

| file | checks |
|---|---|
| `test_embeddings.py` | construction, output shape, batch-1 in eval mode, `x_blocks` declaration, mismatched-layout rejection, wrong-width errors, psd-off and amp-on paths, the custom-embedding contract |
| `test_flows.py` | `log_prob`/`sample` shapes, one-context-many-draws, kwargs reaching zuko, prior-box rejection, `set_bounds` after construction, the custom-flow contract |
| `test_npe.py` | all 36 embedding×flow pairs construct and run, context-mismatch errors, one-x-many-theta broadcast, `pin()` agreeing with the unpinned path, gradients reaching both halves, save/load round-trip |

Add a case whenever a bug is found: the register in
`package_architecture_report.html` lists defects that a construction test
would have caught immediately.



## Broader test files

These tests cover the actual implementations of the above.

| file | catches |
|---|---|---|
| `test_packing.py` | canonical packs x one way, the embedding reads it another — same width, no error, wrong science | 
| `test_config_keys.py` | a config key that silently does nothing (`x_blocks`, `flow_dropout`, `final_stage_flow_dropout` have all been dead at some point) | 
| `test_install.py` | broken subpackage imports, NUL bytes, empty `__all__`, console scripts pointing at a renamed `main()`, a missing runtime dependency | 
| `test_roundtrips.py` | sky reparameterisation, theta normalisation, whitening — the transforms under every posterior | 

### running

```bash
pytest tests -m "not slow"     # the fast suite, every push
pytest tests                   # everything, before a real run
pytest tests/test_packing.py   # after touching canonical.py or embedding.py
```


and in the CI workflow, one can run the slow ones explicitly:

```yaml
      - name: pytest
        run: pytest tests -v --durations=10
      - name: pytest (end to end)
        run: pytest tests -v -m slow
```

