import io
import json

import numpy as np

from spiff.lv2 import _write_row_jsonl


def test_write_row_jsonl_persists_the_emitted_payload() -> None:
    handle = io.StringIO()
    row = {"obsid": "test", "covariance": np.float64(1.25), "flux": np.nan}

    payload = _write_row_jsonl(handle, row)

    assert handle.getvalue() == payload + "\n"
    decoded = json.loads(payload)
    assert decoded["obsid"] == "test"
    assert decoded["covariance"] == 1.25
    assert np.isnan(decoded["flux"])
